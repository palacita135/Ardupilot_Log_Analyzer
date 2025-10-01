#!/usr/bin/env python3
"""
Enhanced ArduPilot/PX4 Log Analyzer
- Supports both ArduPilot (.BIN/.LOG) and PX4 (.ULog) logs
- Full parsing of RC, PM, ESC, servos, sensors, vehicle/firmware metadata
- Stall detection, throttle %, PWM graphs, and more
- Dash GUI with interactive graphs and PDF export
"""
import os
import sys
import argparse
import base64
import re
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pymavlink import mavutil
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.figure_factory as ff
from flask import send_from_directory
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import A3
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import pyulog

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Config
REPORT_DIR = os.path.abspath('reports')
os.makedirs(REPORT_DIR, exist_ok=True)

# Vehicle mode tables
COPTER_MODES = {
    0: 'Stabilize', 1: 'Acro', 2: 'AltHold', 3: 'Auto', 4: 'Guided',
    5: 'Loiter', 6: 'RTL', 7: 'Circle', 9: 'Land', 11: 'Drift',
    12: 'Sport', 13: 'Flip', 14: 'AutoTune', 15: 'PosHold'
}
PLANE_MODES = {
    0: 'Manual', 1: 'Circle', 2: 'Stabilize', 3: 'Training', 4: 'Acro',
    5: 'FBWA', 6: 'FBWB', 7: 'Cruise', 9: 'Auto', 10: 'RTL', 11: 'Loiter'
}
ROVER_MODES = {0: 'Manual', 3: 'Steering', 4: 'Hold', 10: 'Auto', 11: 'RTL'}
SUB_MODES = {0: 'Manual', 3: 'Stabilize', 4: 'DepthHold', 10: 'Auto'}
TRACKER_MODES = {0: 'Manual', 3: 'Stabilize', 10: 'Auto'}
QUADPLANE_MODES = {
    0: 'Manual', 1: 'Circle', 2: 'Stabilize', 3: 'Training', 4: 'Acro',
    5: 'FBWA', 6: 'FBWB', 7: 'Cruise', 9: 'Auto', 10: 'RTL', 11: 'Loiter',
    17: 'QStabilize', 18: 'QHover', 19: 'QLoiter', 20: 'QLand', 21: 'QRTL'
}
VEHICLE_MODE_TABLES = {
    'copter': COPTER_MODES,
    'plane': PLANE_MODES,
    'rover': ROVER_MODES,
    'sub': SUB_MODES,
    'tracker': TRACKER_MODES,
    'quadplane': QUADPLANE_MODES
}

# --- Utilities ---
def find_column(df, patterns):
    if df is None or df.empty:
        return None
    for p in patterns:
        for c in df.columns:
            try:
                if p.lower() in str(c).lower():
                    return c
            except Exception:
                continue
    return None

def safe_float_series(s):
    try:
        return pd.to_numeric(s, errors='coerce')
    except Exception:
        return pd.Series(dtype=float)

def map_custom_mode(val, vehicle='copter'):
    try:
        v = int(val)
        return VEHICLE_MODE_TABLES.get(vehicle, COPTER_MODES).get(v, f'mode_{v}')
    except Exception:
        return str(val)

# --- Parser: Supports ArduPilot and PX4 ---
class LogParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.raw = defaultdict(list)
        self.dfs = {}
        self.vehicle_type = None
        self.firmware = None
        self.firmware_version = None

    def parse(self):
        # Check file signature for ArduPilot (starts with "FMT")
        with open(self.filepath, 'rb') as f:
            header = f.read(32).decode('ascii', errors='ignore')
            if 'FMT' in header:
                return self._parse_ardupilot()
            # Check for PX4 ULog (starts with "UXr")
            elif 'UXr' in header:
                return self._parse_px4()
            else:
                # Fallback to extension-based detection
                if self.filepath.endswith(('.bin', '.log')):
                    return self._parse_ardupilot()
                elif self.filepath.endswith(('.ulg')):
                    return self._parse_px4()
                else:
                    raise RuntimeError("Unsupported log format: Could not detect ArduPilot or PX4 format.")

    def _parse_ardupilot(self):
        try:
            mlog = mavutil.mavlink_connection(self.filepath)
        except Exception as e:
            raise RuntimeError(f"Failed to parse ArduPilot log: {e}. File may be corrupted or not a valid ArduPilot log.")
        while True:
            try:
                m = mlog.recv_match()
            except Exception:
                break
            if m is None:
                break
            mtype = m.get_type()
            if mtype is None or mtype == 'BAD_DATA':
                continue
            try:
                d = m.to_dict()
            except Exception:
                d = {}
                for a in dir(m):
                    if a.startswith('_'):
                        continue
                    try:
                        d[a] = getattr(m, a)
                    except Exception:
                        pass
            t = None
            for tn in ('time_usec', 'time_boot_ms', 'TimeUS', 'time'):
                if tn in d:
                    try:
                        if tn == 'time_boot_ms':
                            t = float(d.get(tn)) / 1000.0
                        elif tn in ('time_usec', 'TimeUS'):
                            t = float(d.get(tn)) / 1e6
                        else:
                            t = float(d.get(tn))
                        break
                    except Exception:
                        t = None
            if t is not None:
                d['_time_s'] = t
            self.raw[mtype.lower()].append(d)
        for k, v in self.raw.items():
            try:
                self.dfs[k] = pd.DataFrame(v)
            except Exception:
                self.dfs[k] = None
        self._normalize_all()
        self._extract_vehicle_firmware()
        return self.dfs

    def _parse_px4(self):
        try:
            ulog = pyulog.ULog(self.filepath)
        except Exception as e:
            raise RuntimeError(f"Failed to parse PX4 log: {e}. File may be corrupted or not a valid PX4 log.")
        for data in ulog.data_list:
            df = pd.DataFrame({f: data.data[f] for f in data.field_names})
            if 'timestamp' in df.columns:
                df['_time_s'] = df['timestamp'] / 1e6  # Convert microseconds to seconds
            self.dfs[data.name] = df
        self._normalize_all()
        self._extract_vehicle_firmware()
        return self.dfs

    def _normalize_all(self):
        self._normalize_gps()
        self._normalize_bat()
        self._normalize_vibe()
        self._normalize_arsp()
        self._normalize_rc()
        self._normalize_rcout()
        self._normalize_pm()
        self._normalize_servo()
        self._normalize_ekf()

    def _normalize_gps(self):
        if 'gps' not in self.dfs or self.dfs['gps'] is None:
            return
        g = self.dfs['gps']
        latc = find_column(g, ['lat', 'latitude'])
        lonc = find_column(g, ['lon', 'lng', 'longitude'])
        sats = find_column(g, ['nsats', 'sats', 'sat'])
        fix = find_column(g, ['fix', 'status', 'gpsfix'])
        rename_map = {}
        if latc and lonc:
            rename_map[latc] = 'lat'
            rename_map[lonc] = 'lon'
        if sats:
            rename_map[sats] = 'NSats'
        if fix:
            rename_map[fix] = 'Fix'
        if rename_map:
            try:
                g = g.rename(columns=rename_map)
            except Exception:
                pass
        try:
            if 'lat' in g.columns and g['lat'].abs().mean() > 1000:
                g['lat'] = g['lat'] / 1e7
            if 'lon' in g.columns and g['lon'].abs().mean() > 1000:
                g['lon'] = g['lon'] / 1e7
        except Exception:
            pass
        self.dfs['gps'] = g

    def _normalize_bat(self):
        if 'bat' not in self.dfs or self.dfs['bat'] is None:
            return
        b = self.dfs['bat']
        vcol = find_column(b, ['volt', 'voltage'])
        if vcol and vcol != 'Volt':
            try:
                b = b.rename(columns={vcol: 'Volt'})
            except Exception:
                pass
        try:
            if 'Volt' in b.columns and b['Volt'].abs().mean() > 100:
                b['Volt'] = b['Volt'] / 1000.0
        except Exception:
            pass
        self.dfs['bat'] = b

    def _normalize_vibe(self):
        if 'vibe' not in self.dfs or self.dfs['vibe'] is None:
            return
        v = self.dfs['vibe']
        rename_map = {}
        for c in v.columns:
            if re.search(r'vibe.*x', c, re.I) or c.lower() == 'vx':
                rename_map[c] = 'VibeX'
            if re.search(r'vibe.*y', c, re.I) or c.lower() == 'vy':
                rename_map[c] = 'VibeY'
            if re.search(r'vibe.*z', c, re.I) or c.lower() == 'vz':
                rename_map[c] = 'VibeZ'
        if rename_map:
            try:
                v = v.rename(columns=rename_map)
            except Exception:
                pass
        self.dfs['vibe'] = v

    def _normalize_arsp(self):
        if 'arsp' not in self.dfs or self.dfs['arsp'] is None:
            return
        a = self.dfs['arsp']
        col = find_column(a, ['airspeed', 'airspeed_m_s', 'airspeed_mps'])
        if col and col != 'Airspeed':
            try:
                a = a.rename(columns={col: 'Airspeed'})
            except Exception:
                pass
        self.dfs['arsp'] = a

    def _normalize_rc(self):
        if 'rc' not in self.dfs or self.dfs['rc'] is None:
            return
        rc = self.dfs['rc']
        # Alias common RC column names
        for i in range(1, 17):
            for suffix in ['', '_in', '_out']:
                col = f'ch{i}{suffix}'
                if col in rc.columns:
                    rc.rename(columns={col: f'Ch{i}'}, inplace=True)
        # Calculate throttle percentage (assuming ch3 is throttle)
        if 'Ch3' in rc.columns:
            rc['ThrottlePercent'] = (rc['Ch3'] - 1000) / 10  # 1000-2000 PWM to 0-100%
        self.dfs['rc'] = rc

    def _normalize_rcout(self):
        for k in list(self.dfs.keys()):
            if k and ('rcout' in k or 'servo' in k or 'actuator' in k):
                df = self.dfs.get(k)
                if df is None:
                    continue
                if '_time_s' not in df.columns:
                    tcol = find_column(df, ['time', 'TimeUS', 'time_boot_ms'])
                    if tcol:
                        try:
                            df['_time_s'] = pd.to_numeric(df[tcol], errors='coerce')
                        except Exception:
                            pass
                self.dfs[k] = df

    def _normalize_pm(self):
        if 'pm' not in self.dfs or self.dfs['pm'] is None:
            return
        pm = self.dfs['pm']
        for col in pm.columns:
            if 'curr' in col.lower():
                if pm[col].abs().mean() > 100:  # Likely in centiampere
                    pm[col] = pm[col] / 100.0
            if 'volt' in col.lower():
                if pm[col].abs().mean() > 100:  # Likely in millivolt
                    pm[col] = pm[col] / 1000.0
        self.dfs['pm'] = pm

    def _normalize_servo(self):
        for k in list(self.dfs.keys()):
            if k and ('servo' in k or 'actuator' in k):
                df = self.dfs.get(k)
                if df is None:
                    continue
                # Alias servo channels
                for i in range(1, 17):
                    for suffix in ['', '_out']:
                        col = f'servo{i}{suffix}'
                        if col in df.columns:
                            df.rename(columns={col: f'Servo{i}'}, inplace=True)
                self.dfs[k] = df

    def _normalize_ekf(self):
        for k in ['ekf', 'ekf_status_report', 'ekf3', 'ekf3_status_report']:
            if k in self.dfs and self.dfs[k] is not None:
                df = self.dfs[k]
                for c in df.columns:
                    if 'err' in c.lower() or 'warning' in c.lower():
                        df.rename(columns={c: f'Error_{c}'}, inplace=True)
                self.dfs[k] = df

    def _extract_vehicle_firmware(self):
        hb = self.dfs.get('heartbeat')
        if hb is not None and not hb.empty:
            try:
                typ = int(hb['type'].iloc[0])
                if typ == 1:
                    self.vehicle_type = 'Fixed Wing (Plane)'
                elif typ == 2:
                    self.vehicle_type = 'Copter (Multicopter)'
                elif typ == 10:
                    self.vehicle_type = 'Rover'
                elif typ == 8:
                    self.vehicle_type = 'Submarine'
                elif typ == 11:
                    self.vehicle_type = 'Tracker'
                elif typ == 14:
                    self.vehicle_type = 'QuadPlane'
                else:
                    self.vehicle_type = 'Unknown'
            except Exception:
                self.vehicle_type = 'Unknown'
            try:
                self.firmware = hb.get('autopilot', 'Unknown').iloc[0]
                self.firmware_version = hb.get('firmware_version', 'Unknown').iloc[0]
            except Exception:
                self.firmware = 'Unknown'
                self.firmware_version = 'Unknown'

# --- Rule Engine ---
class FullRuleEngine:
    def __init__(self, dfs, mode_timeline=None):
        self.dfs = dfs or {}
        self.mode_timeline = mode_timeline
        self.issues = []
        self.evidence = {}
        self.root = None

    def analyze(self):
        self.issues = []
        self.evidence = {}
        self._check_power()
        self._check_gps()
        self._check_vibe_imu()
        self._check_stall()
        self._check_rcout()
        self._check_ekf()
        self._check_failsafe()
        self._check_altitude()
        self._check_ekf3()
        self._check_rc()
        self._check_sensors()
        self._check_speed()
        self._check_tecs()
        self._check_pm()
        self._check_servo()
        self.root = self._classify()
        if self.root:
            self.issues.append({'Subsystem': 'Classifier', 'Severity': 'RootCause', 'Message': self.root})
        self.issues = [it for it in self.issues if isinstance(it, dict)]
        PRIORITY = {
            'Power': 0, 'GPS': 1, 'EKF': 1, 'Vibration': 2, 'IMU': 2, 'Stall': 2.5,
            'Failsafe': 3, 'Mode': 3, 'Servo': 4, 'Actuator': 4, 'Classifier': 10,
            'Altitude': 2, 'EKF3': 1, 'RC': 3, 'Sensors': 2, 'Speed': 2, 'TECS': 1,
            'PM': 0.5
        }
        for it in self.issues:
            it['_priority'] = PRIORITY.get(it.get('Subsystem'), 9)
        for it in self.issues:
            t = it.get('_time_s') or it.get('time')
            if t is not None and self.mode_timeline is not None and not self.mode_timeline.empty:
                try:
                    match = self.mode_timeline[(self.mode_timeline['start'] <= t) & (self.mode_timeline['end'] >= t)]
                    if not match.empty:
                        it['ModeAtEvent'] = match.iloc[0]['mode']
                        it['Message'] = f"{it.get('Message', '')} (mode: {it['ModeAtEvent']})"
                except Exception:
                    pass
        self.issues.sort(key=lambda x: x.get('_priority', 9))
        return self.issues

    def _check_power(self):
        b = self.dfs.get('bat')
        if b is None or b.empty:
            self.evidence['power'] = 'no_data'
            return
        vcol = find_column(b, ['volt', 'voltage']) or 'Volt'
        if vcol not in b.columns:
            self.evidence['power'] = 'no_field'
            return
        v = safe_float_series(b[vcol])
        if v.empty:
            self.evidence['power'] = 'no_data'
            return
        minv = float(v.min())
        maxv = float(v.max())
        if minv < 9.5:
            self.issues.append({
                'Subsystem': 'Power',
                'Severity': 'Critical',
                'Message': f'Low battery voltage (min {minv:.2f}V)',
                '_time_s': b.loc[v.idxmin()].get('_time_s') if '_time_s' in b.columns else None
            })
            self.evidence['power'] = f'low:{minv:.2f}'
            return
        if (v.diff().abs() > 0.2 * max(1.0, v.mean())).any():
            self.issues.append({
                'Subsystem': 'Power',
                'Severity': 'High',
                'Message': 'Sudden voltage drop detected',
                '_time_s': None
            })
            self.evidence['power'] = 'sudden'
            return
        if maxv > 30:
            self.issues.append({
                'Subsystem': 'Power',
                'Severity': 'Warning',
                'Message': f'Over-voltage event (max {maxv:.2f}V)',
                '_time_s': b.loc[v.idxmax()].get('_time_s') if '_time_s' in b.columns else None
            })
            self.evidence['power'] = f'over:{maxv:.2f}'
            return
        self.evidence['power'] = 'ok'

    def _check_pm(self):
        pm = self.dfs.get('pm')
        if pm is None or pm.empty:
            self.evidence['pm'] = 'no_data'
            return
        for col in pm.columns:
            if 'curr' in col.lower():
                s = safe_float_series(pm[col])
                if not s.empty and s.max() > 50:  # Threshold for overload
                    self.issues.append({
                        'Subsystem': 'PM',
                        'Severity': 'Critical',
                        'Message': f'Power overload: {col} max {s.max():.1f}A',
                        '_time_s': pm.loc[s.idxmax()].get('_time_s') if '_time_s' in pm.columns else None
                    })
                    self.evidence['pm'] = 'overload'
                    return
        self.evidence['pm'] = 'ok'

    def _check_servo(self):
        candidate = None
        for k in self.dfs.keys():
            if k and ('servo' in k or 'actuator' in k):
                candidate = k
                break
        if candidate is None:
            self.evidence['servo'] = 'no_data'
            return
        df = self.dfs.get(candidate)
        if df is None or df.empty:
            self.evidence['servo'] = 'no_data'
            return
        stuck = False
        for c in df.columns:
            if re.match(r'^(servo|chan|out|c)\d+', str(c).lower()):
                try:
                    s = pd.to_numeric(df[c], errors='coerce').dropna()
                    if not s.empty and s.max() == s.min():
                        self.issues.append({
                            'Subsystem': 'Servo',
                            'Severity': 'Critical',
                            'Message': f'{c} output appears stuck at {s.max()} PWM',
                            '_time_s': df.iloc[0].get('_time_s') if '_time_s' in df.columns else None
                        })
                        stuck = True
                except Exception:
                    continue
        self.evidence['servo'] = 'stuck' if stuck else 'ok'

    def _check_stall(self):
        arsp = self.dfs.get('arsp')
        gps = self.dfs.get('gps')
        rcout = None
        for k in self.dfs.keys():
            if k and 'rcout' in k:
                rcout = self.dfs[k]
                break
        if arsp is not None and 'Airspeed' in arsp.columns:
            s = safe_float_series(arsp['Airspeed'])
            if not s.empty and (s < 11.0).any():
                idx = int(s.idxmin())
                row = arsp.iloc[idx]
                t = row.get('_time_s')
                throttle_val = None
                if rcout is not None and '_time_s' in rcout.columns:
                    try:
                        nearest = rcout.iloc[(rcout['_time_s'] - t).abs().argsort()[:1]]
                        for c in nearest.columns:
                            if re.match(r'^(chan|servo|out|c)\d+', str(c).lower()):
                                throttle_val = nearest[c].values[0]
                                break
                    except Exception:
                        throttle_val = None
                if throttle_val is None or throttle_val > 1500:
                    self.issues.append({
                        'Subsystem': 'Stall',
                        'Severity': 'High',
                        'Message': f'Stall suspected: low airspeed {row.get("Airspeed"):.1f} m/s',
                        '_time_s': t
                    })
                    self.evidence['stall'] = 'yes'
                    return
        if gps is not None and ('Spd' in gps.columns or 'spd' in gps.columns):
            spd_col = 'Spd' if 'Spd' in gps.columns else 'spd'
            s = safe_float_series(gps[spd_col])
            if not s.empty and (s < 3).any():
                idx = int(s.idxmin())
                row = gps.iloc[idx]
                t = row.get('_time_s')
                throttle_val = None
                if rcout is not None and '_time_s' in rcout.columns:
                    try:
                        nearest = rcout.iloc[(rcout['_time_s'] - t).abs().argsort()[:1]]
                        for c in nearest.columns:
                            if re.match(r'^(chan|servo|out|c)\d+', str(c).lower()):
                                throttle_val = nearest[c].values[0]
                                break
                    except Exception:
                        throttle_val = None
                if throttle_val is None or throttle_val > 1500:
                    self.issues.append({
                        'Subsystem': 'Stall',
                        'Severity': 'High',
                        'Message': f'Stall suspected: low groundspeed {row.get(spd_col):.1f} m/s with throttle',
                        '_time_s': t
                    })
                    self.evidence['stall'] = 'yes'
                    return
        self.evidence['stall'] = 'no'

    def _check_gps(self):
        g = self.dfs.get('gps')
        if g is None or g.empty:
            self.evidence['gps'] = 'no_data'
            return
        sats_col = find_column(g, ['nsat', 'sats', 'sat'])
        fix_col = find_column(g, ['fix', 'status', 'gpsfix'])
        if sats_col and sats_col in g.columns:
            s = safe_float_series(g[sats_col])
            if not s.empty and s.min() < 6:
                self.issues.append({
                    'Subsystem': 'GPS',
                    'Severity': 'Medium',
                    'Message': f'Low satellite count (min {s.min():.0f})',
                    '_time_s': g.loc[s.idxmin()].get('_time_s') if '_time_s' in g.columns else None
                })
                self.evidence['gps'] = 'low_sats'
        if fix_col and fix_col in g.columns:
            fx = safe_float_series(g[fix_col])
            if not fx.empty and (fx < 3).any():
                self.issues.append({
                    'Subsystem': 'GPS',
                    'Severity': 'Critical',
                    'Message': 'GPS lost 3D fix',
                    '_time_s': g.loc[fx.idxmin()].get('_time_s') if '_time_s' in g.columns else None
                })
                self.evidence['gps'] = 'lost_fix'
                return
        if 'gps' not in self.evidence:
            self.evidence['gps'] = 'ok'

    def _check_vibe_imu(self):
        v = self.dfs.get('vibe')
        vib_prob = False
        if v is not None and not v.empty:
            cols = [c for c in v.columns if 'vibe' in c.lower() or c.lower() in ('x', 'y', 'z', 'vibex', 'vibey', 'vibez')]
            for c in cols:
                s = safe_float_series(v[c])
                if not s.empty and s.mean() > 30:
                    self.issues.append({
                        'Subsystem': 'Vibration',
                        'Severity': 'High',
                        'Message': f'High vibration on {c} (mean {s.mean():.1f})',
                        '_time_s': v.iloc[int(s.idxmax())].get('_time_s') if '_time_s' in v.columns else None
                    })
                    vib_prob = True
        imu = self.dfs.get('imu')
        if imu is not None and not imu.empty:
            for c in imu.columns:
                if 'acc' in c.lower() or 'accel' in c.lower():
                    s = safe_float_series(imu[c])
                    if not s.empty and s.abs().max() > 50:
                        self.issues.append({
                            'Subsystem': 'IMU',
                            'Severity': 'High',
                            'Message': f'Acceleration spike on {c} (max {s.abs().max():.1f})',
                            '_time_s': imu.iloc[int(s.abs().idxmax())].get('_time_s') if '_time_s' in imu.columns else None
                        })
                        vib_prob = True
        self.evidence['vibration'] = 'problem' if vib_prob else 'ok'

    def _check_rcout(self):
        candidate = None
        for k in self.dfs.keys():
            if k and ('rcout' in k or 'actuator' in k or 'servo' in k or 'output' in k):
                candidate = k
                break
        if candidate is None:
            self.evidence['servo'] = 'no_data'
            return
        df = self.dfs.get(candidate)
        if df is None or df.empty:
            self.evidence['servo'] = 'no_data'
            return
        stuck = False
        for c in df.columns:
            if re.match(r'^(chan|servo|out|c)\d+', str(c).lower()):
                try:
                    s = pd.to_numeric(df[c], errors='coerce').dropna()
                    if not s.empty and s.max() == s.min():
                        self.issues.append({
                            'Subsystem': 'Servo',
                            'Severity': 'Critical',
                            'Message': f'{c} output appears stuck',
                            '_time_s': df.iloc[0].get('_time_s') if '_time_s' in df.columns else None
                        })
                        stuck = True
                except Exception:
                    continue
        self.evidence['servo'] = 'stuck' if stuck else 'ok'

    def _check_ekf(self):
        ekf = self.dfs.get('ekf') or self.dfs.get('ekf_status_report')
        if ekf is None or ekf.empty:
            self.evidence['ekf'] = 'no_data'
            return
        for c in ekf.columns:
            if 'err' in c.lower() or 'warning' in c.lower() or 'bad' in c.lower():
                try:
                    s = pd.to_numeric(ekf[c], errors='coerce')
                    if not s.empty and (s > 0).any():
                        self.issues.append({
                            'Subsystem': 'EKF',
                            'Severity': 'Critical',
                            'Message': f'EKF error field {c}',
                            '_time_s': ekf.iloc[int(s.idxmax())].get('_time_s') if '_time_s' in ekf.columns else None
                        })
                        self.evidence['ekf'] = 'error'
                        return
                except Exception:
                    continue
        self.evidence['ekf'] = 'ok'

    def _check_failsafe(self):
        st = self.dfs.get('statustext')
        if st is None or st.empty:
            return
        txt_col = find_column(st, ['text', 'message']) or (st.columns[0] if len(st.columns) > 0 else None)
        fails = []
        for _, r in st.iterrows():
            try:
                txt = str(r.get(txt_col, '')).lower()
                if any(tok in txt for tok in ['failsafe', 'fs_', 'fs ', 'rtl', 'land', 'q-land', 'qland', 'assist', 'assisted', 'return to', 'crash']):
                    fails.append(txt)
            except Exception:
                continue
        if fails:
            self.issues.append({
                'Subsystem': 'Failsafe',
                'Severity': 'Critical',
                'Message': '; '.join(fails[:6]),
                '_time_s': st.iloc[0].get('_time_s') if '_time_s' in st.columns else None
            })
            self.evidence['failsafe'] = True

    def _check_altitude(self):
        gps = self.dfs.get('gps')
        if gps is None or gps.empty:
            self.evidence['altitude'] = 'no_data'
            return
        alt_col = find_column(gps, ['alt', 'altitude'])
        if alt_col and alt_col in gps.columns:
            alt = safe_float_series(gps[alt_col])
            if not alt.empty:
                if (alt.diff().abs() > 10).any():
                    self.issues.append({
                        'Subsystem': 'Altitude',
                        'Severity': 'High',
                        'Message': 'Rapid altitude change detected',
                        '_time_s': gps.loc[alt.diff().abs().idxmax()].get('_time_s') if '_time_s' in gps.columns else None
                    })
                    self.evidence['altitude'] = 'rapid_change'
                else:
                    self.evidence['altitude'] = 'ok'
            else:
                self.evidence['altitude'] = 'no_data'
        else:
            self.evidence['altitude'] = 'no_data'

    def _check_ekf3(self):
        ekf3 = self.dfs.get('ekf3') or self.dfs.get('ekf3_status_report')
        if ekf3 is None or ekf3.empty:
            self.evidence['ekf3'] = 'no_data'
            return
        for c in ekf3.columns:
            if 'err' in c.lower() or 'warning' in c.lower() or 'bad' in c.lower():
                try:
                    s = pd.to_numeric(ekf3[c], errors='coerce')
                    if not s.empty and (s > 0).any():
                        self.issues.append({
                            'Subsystem': 'EKF3',
                            'Severity': 'Critical',
                            'Message': f'EKF3 error field {c}',
                            '_time_s': ekf3.iloc[int(s.idxmax())].get('_time_s') if '_time_s' in ekf3.columns else None
                        })
                        self.evidence['ekf3'] = 'error'
                        return
                except Exception:
                    continue
        self.evidence['ekf3'] = 'ok'

    def _check_rc(self):
        rc = self.dfs.get('rc')
        if rc is None or rc.empty:
            self.evidence['rc'] = 'no_data'
            return
        for c in rc.columns:
            if 'in' in c.lower():
                s = safe_float_series(rc[c])
                if not s.empty and (s.diff().abs() > 200).any():
                    self.issues.append({
                        'Subsystem': 'RC',
                        'Severity': 'Medium',
                        'Message': f'Rapid RC input change on {c}',
                        '_time_s': rc.loc[s.diff().abs().idxmax()].get('_time_s') if '_time_s' in rc.columns else None
                    })
                    self.evidence['rc'] = 'rapid_change'
                    return
        self.evidence['rc'] = 'ok'

    def _check_sensors(self):
        imu = self.dfs.get('imu')
        if imu is None or imu.empty:
            self.evidence['sensors'] = 'no_data'
            return
        for c in imu.columns:
            if 'acc' in c.lower() or 'gyro' in c.lower():
                s = safe_float_series(imu[c])
                if not s.empty and s.abs().max() > 100:
                    self.issues.append({
                        'Subsystem': 'Sensors',
                        'Severity': 'High',
                        'Message': f'Sensor spike on {c} (max {s.abs().max():.1f})',
                        '_time_s': imu.iloc[int(s.abs().idxmax())].get('_time_s') if '_time_s' in imu.columns else None
                    })
                    self.evidence['sensors'] = 'spike'
                    return
        self.evidence['sensors'] = 'ok'

    def _check_speed(self):
        gps = self.dfs.get('gps')
        if gps is None or gps.empty:
            self.evidence['speed'] = 'no_data'
            return
        spd_col = find_column(gps, ['spd', 'speed'])
        if spd_col and spd_col in gps.columns:
            spd = safe_float_series(gps[spd_col])
            if not spd.empty and (spd.diff().abs() > 5).any():
                self.issues.append({
                    'Subsystem': 'Speed',
                    'Severity': 'Medium',
                    'Message': 'Rapid speed change detected',
                    '_time_s': gps.loc[spd.diff().abs().idxmax()].get('_time_s') if '_time_s' in gps.columns else None
                })
                self.evidence['speed'] = 'rapid_change'
            else:
                self.evidence['speed'] = 'ok'
        else:
            self.evidence['speed'] = 'no_data'

    def _check_tecs(self):
        tecs = self.dfs.get('tecs')
        if tecs is None or tecs.empty:
            self.evidence['tecs'] = 'no_data'
            return
        for c in tecs.columns:
            if 'err' in c.lower() or 'warning' in c.lower():
                try:
                    s = pd.to_numeric(tecs[c], errors='coerce')
                    if not s.empty and (s > 0).any():
                        self.issues.append({
                            'Subsystem': 'TECS',
                            'Severity': 'Critical',
                            'Message': f'TECS error field {c}',
                            '_time_s': tecs.iloc[int(s.idxmax())].get('_time_s') if '_time_s' in tecs.columns else None
                        })
                        self.evidence['tecs'] = 'error'
                        return
                except Exception:
                    continue
        self.evidence['tecs'] = 'ok'

    def _classify(self):
        ev = self.evidence
        if ev.get('power') and ev['power'] not in ['ok', 'no_data', 'no_field']:
            return 'Power Loss'
        if ev.get('stall') == 'yes':
            return 'Stall / Aerodynamic Failure'
        if ev.get('servo') == 'stuck':
            return 'Actuator/Servo Failure'
        if ev.get('gps') in ['lost_fix', 'low_sats']:
            return 'GPS Failure'
        if ev.get('ekf') == 'error' or ev.get('vibration') == 'problem':
            return 'Navigation/IMU Failure'
        if ev.get('failsafe'):
            return 'Failsafe Trigger'
        if ev.get('altitude') == 'rapid_change':
            return 'Altitude Anomaly'
        if ev.get('ekf3') == 'error':
            return 'EKF3 Error'
        if ev.get('rc') == 'rapid_change':
            return 'RC Input Anomaly'
        if ev.get('sensors') == 'spike':
            return 'Sensor Spike'
        if ev.get('speed') == 'rapid_change':
            return 'Speed Anomaly'
        if ev.get('tecs') == 'error':
            return 'TECS Error'
        positives = [k for k, v in ev.items() if v not in [None, 'ok', 'no_data', 'no_field', 'no']]
        if len(positives) >= 2:
            return 'Multiple/Complex Failure'
        return 'Unknown/Insufficient Evidence'

# --- Mode Timeline ---
def build_mode_timeline(dfs):
    hb = dfs.get('heartbeat')
    if hb is None or hb.empty:
        return pd.DataFrame(columns=['mode', 'start', 'end'])
    vehicle = 'copter'
    if 'type' in hb.columns:
        try:
            typ = int(hb['type'].iloc[0])
            if typ == 1:
                vehicle = 'plane'
            elif typ == 10:
                vehicle = 'rover'
            elif typ == 8:
                vehicle = 'sub'
            elif typ == 11:
                vehicle = 'tracker'
            elif typ == 14:
                vehicle = 'quadplane'
        except Exception:
            pass
    mode_col = find_column(hb, ['custom_mode', 'custom']) or 'custom_mode'
    time_col = find_column(hb, ['_time_s', 'time', 'TimeUS', 'time_boot_ms']) or '_time_s'
    entries = []
    for _, row in hb.iterrows():
        raw = row.get(mode_col)
        t = row.get(time_col) if time_col in hb.columns else row.get('_time_s')
        entries.append({'t': float(t) if t is not None else None, 'raw': raw})
    intervals = []
    prev_mode = None
    start_t = None
    for e in entries:
        mode_name = map_custom_mode(e['raw'], vehicle=vehicle)
        if prev_mode is None:
            prev_mode = mode_name
            start_t = e['t']
        elif mode_name != prev_mode:
            end_t = e['t']
            intervals.append({'mode': prev_mode, 'start': start_t, 'end': end_t})
            prev_mode = mode_name
            start_t = e['t']
    if prev_mode is not None:
        intervals.append({'mode': prev_mode, 'start': start_t, 'end': entries[-1]['t'] if entries else None})
    df = pd.DataFrame(intervals)
    if not df.empty:
        df['start'] = df['start'].fillna(method='ffill').fillna(0)
        df['end'] = df['end'].fillna(df['start'])
    return df

# --- STATUSTEXT Events ---
def extract_statustext_events(dfs):
    st = dfs.get('statustext')
    events = []
    if st is None or st.empty:
        return events
    text_col = find_column(st, ['text', 'message']) or (st.columns[0] if len(st.columns) > 0 else None)
    time_col = find_column(st, ['_time_s', 'time', 'TimeUS', 'time_boot_ms']) or '_time_s'
    for _, row in st.iterrows():
        txt = str(row.get(text_col, '')).strip()
        t = row.get(time_col) if time_col in st.columns else row.get('_time_s')
        if not txt:
            continue
        low = txt.lower()
        if any(tok in low for tok in ['failsafe', 'fs_', 'rtl', 'land', 'q-land', 'qland', 'assist', 'assisted', 'return to', 'crash']):
            events.append({'time': float(t) if t is not None else None, 'text': txt})
    return events

# --- PDF Export ---
def export_report_pdf(issues, dfs, uploaded_name, vehicle_type, firmware, firmware_version):
    issues = [it for it in issues if isinstance(it, dict)]
    if not issues:
        print("Warning: No valid issues to export.")
        return None
    for issue in issues:
        for key, value in issue.items():
            if isinstance(value, np.float64):
                issue[key] = float(value)
    for issue in issues:
        for key, value in issue.items():
            if isinstance(value, (np.float64, float, int)):
                issue[key] = str(value)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base = os.path.splitext(os.path.basename(uploaded_name))[0]
    pdf_name = f"{base}_{ts}.pdf"
    pdf_path = os.path.join(REPORT_DIR, pdf_name)
    issues_sorted = sorted(issues, key=lambda x: x.get('_priority', 9))
    gps = dfs.get('gps')
    start = end = duration = None
    if gps is not None and not gps.empty:
        latc = find_column(gps, ['lat'])
        lonc = find_column(gps, ['lon', 'lng'])
        timec = find_column(gps, ['_time_s', 'time', 'TimeUS']) or '_time_s'
        try:
            if latc and lonc:
                start = (str(float(gps.iloc[0][latc])), str(float(gps.iloc[0][lonc])), str(float(gps.iloc[0].get(timec, np.nan))))
                end = (str(float(gps.iloc[-1][latc])), str(float(gps.iloc[-1][lonc])), str(float(gps.iloc[-1].get(timec, np.nan))))
                if start[2] and end[2]:
                    duration = float(end[2]) - float(start[2])
        except Exception:
            start = end = duration = None
    doc = SimpleDocTemplate(pdf_path, pagesize=A3)
    styles = getSampleStyleSheet()
    elems = [Paragraph('ArduPilot/PX4 Failure Analysis Report', styles['Title']), Spacer(1, 8)]
    elems.append(Paragraph(f'LogFile: {uploaded_name}', styles['Normal']))
    elems.append(Paragraph(f'Generated: {datetime.now().isoformat()}', styles['Normal']))
    elems.append(Paragraph(f'Vehicle Type: {vehicle_type}', styles['Normal']))
    elems.append(Paragraph(f'Firmware: {firmware} {firmware_version}', styles['Normal']))
    if start:
        elems.append(Paragraph(f'Start: lat={start[0]}, lon={start[1]}, t={start[2]}', styles['Normal']))
    if end:
        elems.append(Paragraph(f'End: lat={end[0]}, lon={end[1]}, t={end[2]}', styles['Normal']))
    elems.append(Spacer(1, 8))
    if issues_sorted:
        data = [['Subsystem', 'Severity', 'Message', 'Next Steps']]
        for it in issues_sorted:
            subsys = str(it.get('Subsystem', it.get('subsystem', 'Unknown')))
            sev = str(it.get('Severity', it.get('severity', '')))
            msg = str(it.get('Message', it.get('message', '')))
            mode_at = str(it.get('ModeAtEvent', ''))
            data.append([subsys, sev, f"{msg} (mode: {mode_at})", recommend_next_steps(it)])
        table = Table(data, repeatRows=1, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#d9d9d9')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        elems.append(table)
    else:
        elems.append(Paragraph('No issues detected by automated rules.', styles['Normal']))
    doc.build(elems)
    return pdf_name

# --- Recommend Next Steps ---
def recommend_next_steps(issue):
    if not isinstance(issue, dict):
        return "Further investigation required based on full log review."
    subsystem = str(issue.get("Subsystem", "")).lower()
    if "power" in subsystem:
        return "Check battery health, power module, and wiring."
    if "stall" in subsystem:
        return "Review airspeed sensor calibration, CG, and stall speed margin."
    if "gps" in subsystem:
        return "Inspect GPS antenna, sky visibility, and interference sources."
    if "ekf" in subsystem:
        return "Verify compass calibration, GPS consistency, and vibrations."
    if "imu" in subsystem or "vibration" in subsystem:
        return "Check prop balance, motor mounts, and isolation."
    if "servo" in subsystem or "actuator" in subsystem:
        return "Inspect servo linkage, PWM outputs, and actuator health."
    if "failsafe" in subsystem:
        return "Review failsafe settings, RC link quality, and power redundancy."
    if "mode" in subsystem:
        return "Check mode transitions, mission parameters, and pilot inputs."
    if "altitude" in subsystem:
        return "Check altitude sensor calibration and barometer health."
    if "ekf3" in subsystem:
        return "Review EKF3 parameters and sensor consistency."
    if "rc" in subsystem:
        return "Check RC transmitter calibration and signal quality."
    if "sensors" in subsystem:
        return "Inspect IMU and sensor connections for noise or spikes."
    if "speed" in subsystem:
        return "Verify airspeed sensor calibration and GPS speed consistency."
    if "tecs" in subsystem:
        return "Review TECS configuration and throttle settings."
    if "pm" in subsystem:
        return "Check power module and ESC for overloads or failures."
    return "Further investigation required based on full log review."

# --- Dash App ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

@app.server.route('/reports/<path:filename>')
def serve_reports(filename):
    return send_from_directory(REPORT_DIR, filename, as_attachment=True)

app.layout = html.Div([
    html.H2('Enhanced ArduPilot/PX4 Log Analyzer'),
    dcc.Upload(
        id='upload',
        children=html.Div(['Drag & drop or click to select a BIN/LOG/ULog file']),
        style={'width': '70%', 'height': '70px', 'lineHeight': '70px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '6px', 'textAlign': 'center', 'margin': '12px'},
        multiple=False
    ),
    dcc.Loading(id='loading', children=[html.Div(id='progress')], type='default'),
    html.Div(id='result', style={'marginTop': '12px'}),
    html.Div(id='download_links', style={'marginTop': '8px'})
])

@app.callback(
    [Output('result', 'children'), Output('download_links', 'children'), Output('progress', 'children')],
    [Input('upload', 'contents')],
    [State('upload', 'filename')]
)
def process_upload(contents, filename):
    if contents is None:
        return html.Div('No file uploaded yet.'), html.Div(), ''
    steps = ['Saving upload...']
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
    except Exception as e:
        return html.Div(f'Invalid upload: {e}'), html.Div(), html.Div([html.P(s) for s in steps])
    saved_name = f"{os.path.splitext(filename)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{os.path.splitext(filename)[1]}"
    saved_path = os.path.join(REPORT_DIR, saved_name)
    try:
        with open(saved_path, 'wb') as f:
            f.write(decoded)
        steps.append('File saved')
    except Exception as e:
        return html.Div(f'Failed to save file: {e}'), html.Div(), html.Div([html.P(s) for s in steps])
    steps.append('Parsing log file...')
    parser = LogParser(saved_path)
    try:
        dfs = parser.parse()
        steps.append('Parsing complete')
    except Exception as e:
        return html.Div(f'Parsing failed: {e}'), html.Div(), html.Div([html.P(s) for s in steps])
    steps.append('Building mode timeline and extracting events...')
    try:
        mode_df = build_mode_timeline(dfs)
        events = extract_statustext_events(dfs)
        steps.append('Timeline built')
    except Exception as e:
        mode_df = pd.DataFrame()
        events = []
        steps.append(f'Timeline failed: {e}')
    steps.append('Running rule-based analysis...')
    engine = FullRuleEngine(dfs, mode_timeline=mode_df)
    try:
        issues = engine.analyze()
        steps.append('Analysis complete')
    except Exception as e:
        return html.Div(f'Analysis failed: {e}'), html.Div(), html.Div([html.P(s) for s in steps])
    steps.append('Exporting PDF report...')
    try:
        pdffn = export_report_pdf(issues, dfs, saved_name, parser.vehicle_type, parser.firmware, parser.firmware_version)
        steps.append('Export complete')
    except Exception as e:
        return html.Div(f'Export failed: {e}'), html.Div(), html.Div([html.P(s) for s in steps])

    # --- Generate Visualizations ---
    children = []

    # Flight Info
    flight_info = [
        html.H4('Flight Information'),
        html.P(f"Vehicle Type: {parser.vehicle_type}"),
        html.P(f"Firmware: {parser.firmware} {parser.firmware_version}")
    ]
    children.append(html.Div(flight_info))

    # Flight Path Map
    if 'gps' in dfs and dfs['gps'] is not None:
        gps_df = dfs['gps']
        latc = find_column(gps_df, ['lat', 'latitude'])
        lonc = find_column(gps_df, ['lon', 'lng', 'longitude'])
        if latc and lonc:
            try:
                gps_df = gps_df.rename(columns={latc: 'lat', lonc: 'lon'})
                if gps_df['lat'].abs().mean() > 1000:
                    gps_df['lat'] = gps_df['lat'] / 1e7
                if gps_df['lon'].abs().mean() > 1000:
                    gps_df['lon'] = gps_df['lon'] / 1e7
                fig_map = px.line_mapbox(gps_df, lat='lat', lon='lon', zoom=12, height=420)
                fig_map.update_layout(mapbox_style="open-street-map")
                fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                children.append(html.H4('Flight Path'))
                children.append(dcc.Graph(figure=fig_map))
            except Exception as e:
                children.append(html.P(f"Flight path map error: {e}"))

    # Datetime in HH:MM:SS
    if 'gps' in dfs and dfs['gps'] is not None:
        time_col = find_column(dfs['gps'], ['_time_s', 'time', 'TimeUS'])
        if time_col:
            try:
                start_time = dfs['gps'][time_col].min()
                end_time = dfs['gps'][time_col].max()
                duration = end_time - start_time
                start_dt = datetime.fromtimestamp(0) + timedelta(seconds=start_time)
                end_dt = datetime.fromtimestamp(0) + timedelta(seconds=end_time)
                children.append(html.H4('Flight Duration'))
                children.append(html.P(f"Start: {start_dt.strftime('%H:%M:%S')}"))
                children.append(html.P(f"End: {end_dt.strftime('%H:%M:%S')}"))
                children.append(html.P(f"Duration: {timedelta(seconds=duration)}"))
            except Exception as e:
                children.append(html.P(f"Datetime error: {e}"))

    # RC Analysis
    if 'rc' in dfs and dfs['rc'] is not None:
        rc_df = dfs['rc']
        rc_channels = [c for c in rc_df.columns if c.startswith('Ch') or c.startswith('ch')]
        if rc_channels:
            fig_rc = px.line(rc_df, x='_time_s', y=rc_channels, title='RC Channels (PWM)')
            children.append(html.H4('RC Channels'))
            children.append(dcc.Graph(figure=fig_rc))
        if 'ThrottlePercent' in rc_df.columns:
            fig_throttle = px.line(rc_df, x='_time_s', y='ThrottlePercent', title='Throttle (%)')
            children.append(html.H4('Throttle Percentage'))
            children.append(dcc.Graph(figure=fig_throttle))

    # TECS Analysis
    if 'tecs' in dfs and dfs['tecs'] is not None:
        tecs_df = dfs['tecs']
        tecs_cols = [c for c in tecs_df.columns if 'err' not in c.lower()]
        if tecs_cols:
            fig_tecs = px.line(tecs_df, x='_time_s', y=tecs_cols, title='TECS Telemetry')
            children.append(html.H4('TECS Telemetry'))
            children.append(dcc.Graph(figure=fig_tecs))

    # Servo Analysis
    servo_df = None
    for k in dfs.keys():
        if k and ('servo' in k or 'actuator' in k):
            servo_df = dfs[k]
            break
    if servo_df is not None:
        servo_channels = [c for c in servo_df.columns if re.match(r'^(servo|chan|out|c)\d+', str(c).lower())]
        if servo_channels:
            fig_servo = px.line(servo_df, x='_time_s', y=servo_channels, title='Servo PWM Outputs')
            children.append(html.H4('Servo PWM Outputs'))
            children.append(dcc.Graph(figure=fig_servo))

    # Sensor Analysis (IMU)
    if 'imu' in dfs and dfs['imu'] is not None:
        imu_df = dfs['imu']
        imu_cols = [c for c in imu_df.columns if 'acc' in c.lower() or 'gyro' in c.lower()]
        if imu_cols:
            fig_imu = px.line(imu_df, x='_time_s', y=imu_cols, title='IMU Telemetry (Accel/Gyro)')
            children.append(html.H4('IMU Telemetry'))
            children.append(dcc.Graph(figure=fig_imu))

    # Altitude Analysis
    if 'gps' in dfs and dfs['gps'] is not None:
        alt_col = find_column(dfs['gps'], ['alt', 'altitude'])
        if alt_col:
            fig_alt = px.line(dfs['gps'], x='_time_s', y=alt_col, title='Altitude Over Time')
            children.append(html.H4('Altitude Over Time'))
            children.append(dcc.Graph(figure=fig_alt))

    # EKF/EKF3 Analysis
    for ekf_key in ['ekf', 'ekf3', 'ekf_status_report', 'ekf3_status_report']:
        if ekf_key in dfs and dfs[ekf_key] is not None:
            ekf_df = dfs[ekf_key]
            ekf_cols = [c for c in ekf_df.columns if 'err' in c.lower() or 'warning' in c.lower()]
            if ekf_cols:
                fig_ekf = px.line(ekf_df, x='_time_s', y=ekf_cols, title=f'{ekf_key.upper()} Errors')
                children.append(html.H4(f'{ekf_key.upper()} Errors'))
                children.append(dcc.Graph(figure=fig_ekf))

    # PM (Power Module) Analysis
    if 'pm' in dfs and dfs['pm'] is not None:
        pm_df = dfs['pm']
        pm_cols = [c for c in pm_df.columns if 'curr' in c.lower() or 'volt' in c.lower()]
        if pm_cols:
            fig_pm = px.line(pm_df, x='_time_s', y=pm_cols, title='Power Module Telemetry')
            children.append(html.H4('Power Module Telemetry'))
            children.append(dcc.Graph(figure=fig_pm))

    # Speed Analysis
    if 'gps' in dfs and dfs['gps'] is not None:
        spd_col = find_column(dfs['gps'], ['spd', 'speed'])
        if spd_col:
            fig_spd = px.line(dfs['gps'], x='_time_s', y=spd_col, title='Speed Over Time')
            children.append(html.H4('Speed Over Time'))
            children.append(dcc.Graph(figure=fig_spd))

    # Mode Timeline
    if not mode_df.empty:
        try:
            base_t = datetime.utcnow()
            md = mode_df.copy()
            md['start'] = md['start'].fillna(0)
            md['end'] = md['end'].fillna(md['start'] + 1)
            md['start_dt'] = md['start'].apply(lambda s: base_t + timedelta(seconds=float(s) if s is not None else 0))
            md['end_dt'] = md['end'].apply(lambda s: base_t + timedelta(seconds=float(s) if s is not None else 0))
            fig_gantt = px.timeline(md, x_start='start_dt', x_end='end_dt', y='mode', color='mode', height=300)
            for ev in events:
                if ev.get('time') is None:
                    continue
                ev_dt = base_t + timedelta(seconds=float(ev['time']))
                fig_gantt.add_vline(x=ev_dt, line=dict(color='red', width=2, dash='dash'))
                fig_gantt.add_annotation(x=ev_dt, y=0.5, text=ev.get('text'), showarrow=True, arrowhead=1)
            fig_gantt.update_yaxes(autorange='reversed')
            children.append(html.H4('Mode Timeline'))
            children.append(dcc.Graph(figure=fig_gantt))
        except Exception as e:
            children.append(html.P(f"Mode timeline error: {e}"))

    # Issues Summary
    if issues:
        summary_children = [
            html.H4('Detected Issues'),
            html.Ul([html.Li(f"{it.get('Subsystem', '')} [{it.get('Severity', '')}] — {it.get('Message', '')}") for it in issues])
        ]
        children.append(html.Div(summary_children))

    downloads = html.Div([html.A('Download PDF', href=f'/reports/{pdffn}', target='_blank')])
    return html.Div(children), downloads, html.Div([html.P(s) for s in steps])


# --- CLI Mode ---
def run_file_mode(path):
    parser = LogParser(path)
    dfs = parser.parse()
    mode_df = build_mode_timeline(dfs)
    engine = FullRuleEngine(dfs, mode_timeline=mode_df)
    issues = engine.analyze()
    pdffn = export_report_pdf(issues, dfs, os.path.basename(path), parser.vehicle_type, parser.firmware, parser.firmware_version)
    print(f'Report: {os.path.join(REPORT_DIR, pdffn)}')

# --- Main ---
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['dash', 'file'], required=True)
    ap.add_argument('--log', help='path to .BIN/.LOG/.ULog for file mode')
    args = ap.parse_args()
    if args.mode == 'dash':
        app.run(debug=True)
    else:
        if not args.log:
            print('--log required for file mode')
            sys.exit(1)
        run_file_mode(args.log)
