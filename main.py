import sys
import serial # For actual VCU, install with: pip install pyserial
import time
import random
import csv
import os
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGridLayout, QMessageBox, QFrame, QCheckBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QColor, QPalette

import pyqtgraph as pg # For plotting, install with: pip install pyqtgraph
import numpy as np # For slope calculation, install with: pip install numpy

# --- Global Configuration ---
# Set to True for mock data, False for actual Serial Port
USE_MOCK_DATA = True

# Serial Port Configuration (if USE_MOCK_DATA is False)
# IMPORTANT: Adjust these for your actual VCU/Virtual Serial Port setup
SERIAL_PORT = 'COM4' # On Linux/macOS, it might be '/dev/pts/1' or '/dev/ttyUSB0'
BAUD_RATE = 115200
SERIAL_READ_TIMEOUT = 1 # seconds

UPDATE_INTERVAL_MS = 100 # How often the data reader attempts to read (faster than display update)
DISPLAY_UPDATE_INTERVAL_MS = 500 # How often the GUI updates with new data

MAX_GRAPH_POINTS = 200 # Max data points to show on time-series graphs
CSV_FILE = "evr_data_log.csv"

# --- Data Caching and Global State ---
mock_vcu_readings = {
    "voltage": 48.0,
    "motor_temp": 60.0,
    "throttle": 0,
    "torque": 0.0,
    "current": 0.0,
    "g_force_x": 0.0,
    "g_force_y": 0.0,
    "soc": 100.0,
    "soc_prev": 100.0 # To track previous SoC for trend calculation
}

# --- Helper Functions (Re-used from Flask Logic) ---

def generate_mock_data(prev_data):
    # Store prev_data["soc"] before modifying for accurate trend calculation
    prev_soc = prev_data["soc"]

    voltage = prev_data["voltage"] + random.uniform(-0.5, 0.5)
    voltage = max(40.0, min(60.0, voltage))

    motor_temp = prev_data["motor_temp"] + random.uniform(-1.0, 2.0)
    motor_temp = max(50.0, min(100.0, motor_temp))

    throttle = prev_data["throttle"] + random.randint(-5, 5)
    throttle = max(0, min(100, throttle))

    torque = prev_data["torque"] + random.uniform(-2.0, 3.0)
    torque = max(0.0, min(60.0, torque))

    current = prev_data["current"] + random.uniform(-3.0, 4.0)
    current = max(0.0, min(90.0, current))

    g_force_x = prev_data["g_force_x"] + random.uniform(-0.1, 0.1)
    g_force_x = max(-2.0, min(2.0, g_force_x))

    g_force_y = prev_data["g_force_y"] + random.uniform(-0.1, 0.1)
    g_force_y = max(-2.0, min(2.0, g_force_y))

    soc = prev_data["soc"] - random.uniform(0.01, 0.05)
    soc = max(0.0, soc)

    return {
        "voltage": round(voltage, 2),
        "motor_temp": round(motor_temp, 1),
        "throttle": throttle,
        "torque": round(torque, 2),
        "current": round(current, 2),
        "g_force_x": round(g_force_x, 2),
        "g_force_y": round(g_force_y, 2),
        "soc": round(soc, 2),
        "soc_prev": prev_soc # Pass previous SoC for status/trend
    }

def compute_status_trend(key, value, prev_value=None):
    status = "normal"
    trend = "steady"

    if key == "soc":
        if value < 20: status = "critical"
        elif value < 40: status = "warning"
    elif key == "motor_temp":
        if value > 90: status = "critical"
        elif value > 75: status = "warning"
    elif key == "voltage":
        if value < 45 or value > 55: status = "critical"
        elif value < 47 or value > 53: status = "warning"
    elif key == "current":
        if value > 80: status = "critical"
        elif value > 60: status = "warning"
    elif key == "throttle":
        pass # No specific critical/warning for throttle
    elif key == "torque":
        pass # No specific critical/warning for torque
    elif key == "g_force_x" or key == "g_force_y":
        if abs(value) > 1.5: status = "warning"
        if abs(value) > 2.5: status = "critical" # Added critical for G-Force

    if prev_value is not None:
        if value > prev_value:
            trend = "up"
        elif value < prev_value:
            trend = "down"
        else:
            trend = "steady"
    return status, trend

def calculate_slope(x_data, y_data, num_points=20):
    """
    Calculates the slope (rate of change) from the last `num_points` of data.
    Uses linear regression to find the best fit line.
    Returns slope in units of Y per unit of X.
    """
    if len(x_data) < 2 or len(y_data) < 2:
        return 0.0 # Not enough data points to calculate slope

    points_to_use = min(num_points, len(x_data))
    
    # x_data is Unix timestamp in milliseconds, convert to relative seconds
    # Make sure x_seconds is relative to the start of the slice for polyfit accuracy
    x_relative = (np.array(x_data[-points_to_use:]) - x_data[-points_to_use]) / 1000.0
    y_values = np.array(y_data[-points_to_use:])

    if len(np.unique(x_relative)) < 2:
        return 0.0

    slope, _ = np.polyfit(x_relative, y_values, 1)
    
    return float(slope)

# --- VCU Data Reader Worker (Runs in a separate thread) ---

class VcuDataReader(QObject):
    data_received = pyqtSignal(dict) # Signal to send processed VCU data to main thread
    
    def __init__(self, use_mock, serial_port=None, baud_rate=None, update_interval_ms=100):
        super().__init__()
        self._running = True
        self.use_mock = use_mock
        self.serial_port_name = serial_port
        self.baud_rate = baud_rate
        self.update_interval_ms = update_interval_ms
        self.ser = None # Serial object
        self.mock_data = mock_vcu_readings.copy() # Local copy of mock data state

    def run(self):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] VCU Data Reader thread started (Mock: {self.use_mock}).")
        while self._running:
            try:
                latest_data = {}
                if self.use_mock:
                    # Generate mock data
                    self.mock_data = generate_mock_data(self.mock_data)
                    latest_data = self.mock_data.copy()
                else:
                    # Read from actual serial port
                    if self.ser is None or not self.ser.is_open:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Attempting to open serial port {self.serial_port_name}...")
                        self.ser = serial.Serial(self.serial_port_name, self.baud_rate, timeout=SERIAL_READ_TIMEOUT)
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Serial port {self.serial_port_name} opened successfully.")
                    
                    line = self.ser.readline().decode('utf-8').strip()
                    if line:
                        # Parse VCU data (similar to Flask's get_data)
                        try:
                            parts = line.split(',')
                            if len(parts) == 8: # Expected 8 values
                                latest_data = {
                                    "motor_temp": float(parts[0]),
                                    "voltage": float(parts[1]),
                                    "current": float(parts[2]),
                                    "throttle": int(parts[3]),
                                    "torque": float(parts[4]),
                                    "g_force_x": float(parts[5]),
                                    "g_force_y": float(parts[6]),
                                    "soc": float(parts[7]),
                                    "soc_prev": self.mock_data["soc"] # Use previous mock_data's soc for trend if not mocked
                                }
                                # Update mock_data's soc for next iteration's soc_prev
                                self.mock_data["soc"] = latest_data["soc"]
                            else:
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] Skipping malformed VCU data: '{line}' (incorrect parts count)")
                                continue # Skip emitting if data is bad
                        except (ValueError, IndexError) as e:
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error parsing VCU data '{line}': {e}")
                            continue # Skip emitting if data is bad
                    else:
                        # No data read within timeout, just continue loop
                        # print(f"[{datetime.now().strftime('%H:%M:%S')}] No data from serial port.")
                        time.sleep(self.update_interval_ms / 1000.0)
                        continue # Skip emitting if no new data

                if latest_data:
                    # Emit data with current timestamp for consistency
                    data_with_timestamp = {
                        "timestamp_ms": int(time.time() * 1000),
                        "values": latest_data
                    }
                    self.data_received.emit(data_with_timestamp)

            except serial.SerialException as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Serial port error: {e}. Retrying in 5 seconds...")
                if self.ser and self.ser.is_open:
                    self.ser.close()
                self.ser = None
                time.sleep(5)
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Unexpected error in VCU read loop: {e}. Retrying in 1 second...")
                time.sleep(1)

            time.sleep(self.update_interval_ms / 1000.0) # Control loop speed

    def stop(self):
        self._running = False
        if self.ser and self.ser.is_open:
            self.ser.close()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Serial port closed.")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] VCU Data Reader thread stopped.")


# --- Main Dashboard Application Window ---

class DashboardWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EVR Racing Dashboard")
        self.setGeometry(100, 100, 1400, 900) # Increased initial window size
        self.setMinimumSize(1000, 700) # Increased minimum size

        self.session_start_time = time.time()
        self.last_lap_reset_time = time.time()
        self.best_lap_time = float('inf')

        # Data caches for graphs (Unix timestamps in milliseconds)
        self.time_data = []
        self.motor_temp_data = []
        self.voltage_data = []
        self.current_data = []
        self.torque_data = []
        self.throttle_data = []
        self.g_force_x_data = []
        self.g_force_y_data = []
        
        # Clear CSV file on startup for fresh logs
        if os.path.exists(CSV_FILE):
            os.remove(CSV_FILE)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Cleared existing CSV log: {CSV_FILE}")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] CSV log will be created: {CSV_FILE}")

        self.initUI()
        self.start_data_acquisition()

    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(15)

        # --- Header Section ---
        header_layout = QHBoxLayout()
        header_layout.setSpacing(20)

        app_title = QLabel("<h1>EVR Racing Dashboard</h1>")
        app_title.setStyleSheet("color: #fff;")
        header_layout.addWidget(app_title, alignment=Qt.AlignLeft)

        time_display_layout = QHBoxLayout()
        time_display_layout.setSpacing(10)

        # Current Time
        self.current_time_label = QLabel("--:--:--")
        self.current_time_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.current_time_label.setStyleSheet("background-color: #2c2c2c; padding: 8px 15px; border-radius: 5px; border: 1px solid #444; color: #fff;")
        time_display_layout.addWidget(self.current_time_label)

        # Session Duration
        self.session_duration_label = QLabel("Duration: 00:00:00")
        self.session_duration_label.setFont(QFont("Segoe UI", 16))
        self.session_duration_label.setStyleSheet("background-color: #2c2c2c; padding: 8px 15px; border-radius: 5px; border: 1px solid #444; color: #ccc;")
        time_display_layout.addWidget(self.session_duration_label)
        
        header_layout.addStretch(1) # Pushes time displays to the right
        header_layout.addLayout(time_display_layout)
        self.main_layout.addLayout(header_layout)

        # --- KPI Grid ---
        self.kpi_grid_layout = QGridLayout()
        self.kpi_grid_layout.setSpacing(10)
        self.kpi_labels = {} # Store QLabel objects for easy update

        # Define KPI order and properties
        self.kpi_config = {
            "Lap Time": {"unit": "", "font_size": 32, "bold": True},
            "SoC": {"unit": "%", "font_size": 32, "bold": True},
            "Motor Temp": {"unit": "°C", "font_size": 32, "bold": True},
            "Motor Temp Slope": {"unit": "°C/s", "font_size": 24, "bold": False, "color_override": "#FFC107"}, # Added Slope KPI with specific color
            "Voltage": {"unit": "V", "font_size": 32, "bold": True},
            "Current": {"unit": "A", "font_size": 32, "bold": True},
            "Throttle": {"unit": "%", "font_size": 32, "bold": True},
            "Torque": {"unit": "Nm", "font_size": 32, "bold": True},
            "G-Force X": {"unit": "g", "font_size": 32, "bold": True},
            "G-Force Y": {"unit": "g", "font_size": 32, "bold": True}
        }

        row, col = 0, 0
        for kpi_name, kpi_props in self.kpi_config.items():
            kpi_frame = QWidget()
            kpi_frame.setStyleSheet("""
                QWidget {
                    background-color: #2c2c2c;
                    border: 1px solid #444;
                    border-radius: 5px;
                }
            """)
            kpi_frame_layout = QVBoxLayout(kpi_frame)
            kpi_frame_layout.setContentsMargins(15, 15, 15, 15)
            kpi_frame_layout.setSpacing(5)
            # kpi_frame.setFixedSize(200, 140) # REMOVED: To allow flexible sizing for KPI frames

            title_label = QLabel(kpi_name)
            title_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
            title_label.setStyleSheet("color: #aaa;")
            title_label.setAlignment(Qt.AlignCenter)
            kpi_frame_layout.addWidget(title_label)

            value_label = QLabel("--.- " + kpi_props["unit"])
            font_weight = QFont.Bold if kpi_props["bold"] else QFont.Normal
            value_label.setFont(QFont("Segoe UI", kpi_props["font_size"], font_weight))
            
            # Apply specific color override if defined (e.g., for Motor Temp Slope)
            if "color_override" in kpi_props:
                value_label.setStyleSheet(f"color: {kpi_props['color_override']};")
            else:
                value_label.setStyleSheet(f"color: white;") # Default color, will be updated by status
            
            # Set alignment for value label
            # If it's Lap Time, allow text wrapping by setting a suitable alignment for multi-line
            if kpi_name == "Lap Time":
                value_label.setAlignment(Qt.AlignCenter) # Set alignment first
                value_label.setWordWrap(True)            # Then enable word wrap
            else:
                value_label.setAlignment(Qt.AlignCenter)
            
            self.kpi_labels[kpi_name] = value_label # Store reference
            kpi_frame_layout.addWidget(value_label)
            
            kpi_frame_layout.addStretch(1) # Push content to top

            self.kpi_grid_layout.addWidget(kpi_frame, row, col)
            col += 1
            if col > 3: # 4 columns per row
                col = 0
                row += 1
        
        # Add stretch to the columns of the KPI grid layout to allow them to expand
        for i in range(4): # Assuming 4 columns in your KPI grid
            self.kpi_grid_layout.setColumnStretch(i, 1)

        self.main_layout.addLayout(self.kpi_grid_layout)

        # --- Graph Section ---
        self.graph_section_layout = QGridLayout()
        self.graph_section_layout.setSpacing(15)
        # ให้กราฟมีพื้นที่ขยายมากขึ้น (Increased stretch factors for graphs)
        self.graph_section_layout.setRowStretch(0, 3) # More space for first row of graphs
        self.graph_section_layout.setRowStretch(1, 3) # More space for second row of graphs
        self.graph_section_layout.setColumnStretch(0, 1)
        self.graph_section_layout.setColumnStretch(1, 1)


        # Common plot configuration
        pg.setConfigOption('background', '#1a1a1a')
        pg.setConfigOption('foreground', '#eee')
        pg.setConfigOption('leftButtonPan', True) # Enable pan with left mouse click (default PyQtGraph behavior)
        pg.setConfigOption('antialias', True) # Smoother lines

        # Motor Temp Graph
        self.motor_temp_plot_container, self.motor_temp_plot = self._create_graph_widget("Motor Temperature (°C)", "Temperature (°C)", "Time (s)")
        self.temp_curve = self.motor_temp_plot.plot(pen=pg.mkPen(color=(255, 99, 132), width=2))
        self.graph_section_layout.addWidget(self.motor_temp_plot_container, 0, 0)
        
        # Voltage/Current Graph
        self.voltage_current_plot_container, self.voltage_current_plot = self._create_graph_widget("Voltage (V) / Current (A)", "Value", "Time (s)",
                                                                                                      curve_names=["Voltage (V)", "Current (A)"],
                                                                                                      curve_colors=[(54, 162, 235), (75, 192, 192)])
        self.voltage_curve = self.voltage_current_plot.plot(pen=pg.mkPen(color=(54, 162, 235), width=2), name="Voltage (V)")
        self.current_curve = self.voltage_current_plot.plot(pen=pg.mkPen(color=(75, 192, 192), width=2), name="Current (A)")
        self.graph_section_layout.addWidget(self.voltage_current_plot_container, 0, 1)

        # G-Forces Graph
        self.g_force_plot_container, self.g_force_plot = self._create_graph_widget("G-Forces (g)", "G-Force", "Time (s)",
                                                                                      curve_names=["G-Force X", "G-Force Y"],
                                                                                      curve_colors=[(255, 205, 86), (153, 102, 255)])
        self.g_force_x_curve = self.g_force_plot.plot(pen=pg.mkPen(color=(255, 205, 86), width=2), name="G-Force X")
        self.g_force_y_curve = self.g_force_plot.plot(pen=pg.mkPen(color=(153, 102, 255), width=2), name="G-Force Y")
        self.graph_section_layout.addWidget(self.g_force_plot_container, 1, 0)
        
       # Throttle vs. Torque Correlation Graph 
        self.correlation_plot_container, self.correlation_plot = self._create_graph_widget(
            "Throttle vs. Torque", "Torque (Nm)", "Throttle (%)", is_scatter=True,
            curve_names=["Historical Data", "Current Point"],
            curve_colors=[(132, 112, 255), (132, 112, 255)] 
        )
        # Historical scatter (fainter, smaller)
        self.throttle_torque_historical_scatter = self.correlation_plot.plot(
            symbol='o', symbolSize=6, pen=None, brush=pg.mkBrush(132, 112, 255), 
            name="Historical Data"
        )
        # Current point scatter (brighter, larger, RED)
        self.throttle_torque_current_scatter = self.correlation_plot.plot(
            symbol='o', symbolSize=10, pen=pg.mkPen(color=(132, 112, 255), width=2),
            brush=pg.mkBrush(132, 112, 255), # Solid RED fill
            name="Current Point"
        )

        # Set fixed range for correlation graph as per original design
        self.correlation_plot.setXRange(0, 100, padding=0.05)
        self.correlation_plot.setYRange(0, 60, padding=0.05)
        self.graph_section_layout.addWidget(self.correlation_plot_container, 1, 1)


        self.main_layout.addLayout(self.graph_section_layout)

        # --- Controls Section ---
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 15, 0, 0)
        controls_layout.setSpacing(10)
        controls_layout.addStretch(1) # Push content to center/right

        contact_label = QLabel("Contact Developer: sorasaklaopraphaiphan@gmail.com")
        contact_label.setStyleSheet("color: #aaa; font-size: 14px;")
        controls_layout.addWidget(contact_label)

        controls_layout.addStretch(1)
        self.main_layout.addLayout(controls_layout)

    def _create_graph_widget(self, title, y_axis_label, x_axis_label, is_scatter=False, curve_names=None, curve_colors=None):
        # Create a QWidget to act as a container for the plot and its button/checkboxes
        container_widget = QWidget()
        container_widget.setStyleSheet("""
            QWidget {
                background-color: #1a1a1a;
                border: 1px solid #333;
                border-radius: 5px;
                padding: 0px;
            }
        """)
        
        plot_layout = QVBoxLayout(container_widget) # Set container_widget as parent
        plot_layout.setContentsMargins(0,0,0,0)
        plot_layout.setSpacing(0)

        plot_widget = pg.PlotWidget()
        plot_widget.setTitle(title)
        plot_widget.setLabel('left', y_axis_label)
        plot_widget.setLabel('bottom', x_axis_label)
        plot_widget.showGrid(x=True, y=True)
        plot_widget.setMouseEnabled(x=True, y=True)
        
        plot_layout.addWidget(plot_widget)
        
        # Bottom controls layout for reset button and checkboxes
        bottom_controls_layout = QHBoxLayout()
        bottom_controls_layout.setContentsMargins(5, 5, 5, 5) # Add some padding around controls
        bottom_controls_layout.setSpacing(10)

        # Reset Zoom button (now on left, with icon)
        reset_btn = QPushButton("↻") # Unicode circular arrows refresh symbol
        reset_btn.setFont(QFont("Arial", 16)) # Larger font for icon
        reset_btn.setFixedSize(30, 30) # Fixed size for a button icon
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(68, 68, 68, 0.7);
                color: #aaa;
                border: 1px solid #555;
                border-radius: 15px; /* Make it circular */
                padding: 0px; /* Adjust padding for icon */
            }
            QPushButton:hover {
                background-color: rgba(85, 85, 85, 0.9);
                color: white;
            }
        """)
        # For scatter plot, we do NOT want autoRange on reset, we want fixed range
        reset_btn.clicked.connect(lambda: self._reset_graph_zoom(plot_widget, is_scatter))
        bottom_controls_layout.addWidget(reset_btn, alignment=Qt.AlignBottom | Qt.AlignLeft)
        
        # Add checkboxes for multi-curve graphs
        if curve_names and curve_colors:
            plot_widget.checkboxes = [] # Initialize list to store (checkbox, name) tuples
            for i, name in enumerate(curve_names):
                checkbox = QCheckBox(name)
                checkbox.setChecked(True) # Default to visible
                
                # Set checkbox color to match curve color for better UX
                color_rgb = curve_colors[i]
                # Check if alpha (transparency) is provided, if so convert to rgba
                if len(color_rgb) == 4:
                    checkbox.setStyleSheet(f"color: rgba({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]}, {color_rgb[3]});")
                else:
                    checkbox.setStyleSheet(f"color: rgb({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]});")
                
                bottom_controls_layout.addWidget(checkbox)
                plot_widget.checkboxes.append((checkbox, name)) # Store for later connection

        bottom_controls_layout.addStretch(1) # Push button/checkboxes to the left/center
        plot_layout.addLayout(bottom_controls_layout)

        return container_widget, plot_widget 

    def _reset_graph_zoom(self, plot_widget_instance, is_scatter=False):
        # Time-series graphs autoRange
        if not is_scatter:
            plot_widget_instance.autoRange()
        # Scatter plot reverts to predefined fixed ranges
        else:
            plot_widget_instance.setXRange(0, 100, padding=0.05)
            plot_widget_instance.setYRange(0, 60, padding=0.05)

    def _connect_plot_checkboxes(self, plot_widget):
        """Connects checkboxes to their corresponding plot items after curves are created."""
        if hasattr(plot_widget, 'checkboxes'):
            for checkbox, curve_name in plot_widget.checkboxes:
                # Find the actual PlotDataItem by its 'name'
                for item in plot_widget.listDataItems():
                    if hasattr(item, 'name') and item.name() == curve_name: # Check if item has name attribute
                        # Connect the checkbox toggled signal to the item's setVisible slot
                        checkbox.toggled.connect(item.setVisible)
                        break

    def start_data_acquisition(self):
        # Create a QThread instance
        self.data_thread = QThread()
        # Create a worker object (VcuDataReader)
        self.vcu_reader = VcuDataReader(
            USE_MOCK_DATA,
            SERIAL_PORT if not USE_MOCK_DATA else None,
            BAUD_RATE if not USE_MOCK_DATA else None,
            UPDATE_INTERVAL_MS
        )
        # Move the worker object to the thread
        self.vcu_reader.moveToThread(self.data_thread)

        # Connect signals and slots
        self.data_thread.started.connect(self.vcu_reader.run)
        self.vcu_reader.data_received.connect(self.update_dashboard)

        # Handle thread termination (important for clean shutdown)
        self.data_thread.finished.connect(self.data_thread.deleteLater)
        self.vcu_reader.destroyed.connect(self.data_thread.quit) # Stop thread when worker is destroyed

        # Start the thread
        self.data_thread.start()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Data acquisition thread started.")

        # Ensure worker stops when app quits
        QApplication.instance().aboutToQuit.connect(self.vcu_reader.stop)
        QApplication.instance().aboutToQuit.connect(self.data_thread.quit)
        QApplication.instance().aboutToQuit.connect(self.data_thread.wait)

        # Connect the checkboxes to their plot items after all plots are created
        # This must be called after self.voltage_curve, self.current_curve, etc., are defined
        self._connect_plot_checkboxes(self.voltage_current_plot)
        self._connect_plot_checkboxes(self.g_force_plot)
        self._connect_plot_checkboxes(self.correlation_plot) # Connect for Throttle vs. Torque too


    def update_dashboard(self, data_packet):
        # data_packet is {"timestamp_ms": int, "values": dict}
        current_time_ms = data_packet["timestamp_ms"]
        current_vcu_data = data_packet["values"]

        # --- Update Time Display ---
        self.update_time_display(current_time_ms)

        # --- Calculate Lap Time ---
        current_time_seconds = current_time_ms / 1000.0
        current_lap_duration = current_time_seconds - self.last_lap_reset_time

        lap_time_display = ""
        # Simulate lap completion (e.g., every 60 seconds)
        if current_lap_duration >= 60.0: # Simulate a lap every 60 seconds
            completed_lap_duration = current_lap_duration
            if completed_lap_duration < self.best_lap_time:
                self.best_lap_time = completed_lap_duration
            
            lap_time_display = f"{int(completed_lap_duration // 60):02d}:{int(completed_lap_duration % 60):02d}:{int((completed_lap_duration * 1000) % 1000):03d}"
            if self.best_lap_time != float('inf'):
                best_lap_time_str = f"{int(self.best_lap_time // 60):02d}:{int(self.best_lap_time % 60):02d}:{int((self.best_lap_time * 1000) % 1000):03d}"
                lap_time_display += f"\n(Best: {best_lap_time_str})" # Use newline for best time to give more space
            
            self.last_lap_reset_time = current_time_seconds # Reset for new lap
        else:
            # Always show the current lap time, and best if available
            current_lap_str = f"{int(current_lap_duration // 60):02d}:{int(current_lap_duration % 60):02d}:{int((current_lap_duration * 1000) % 1000):03d}"
            lap_time_display = current_lap_str
            if self.best_lap_time != float('inf'):
                best_lap_time_str = f"{int(self.best_lap_time // 60):02d}:{int(self.best_lap_time % 60):02d}:{int((self.best_lap_time * 1000) % 1000):03d}"
                lap_time_display += f"\n(Best: {best_lap_time_str})" # Use newline for best time to give more space

        # --- Store Time Series Data ---
        self.time_data.append(current_time_ms)
        self.motor_temp_data.append(current_vcu_data["motor_temp"])
        self.voltage_data.append(current_vcu_data["voltage"])
        self.current_data.append(current_vcu_data["current"])
        self.torque_data.append(current_vcu_data["torque"])
        self.throttle_data.append(current_vcu_data["throttle"])
        self.g_force_x_data.append(current_vcu_data["g_force_x"])
        self.g_force_y_data.append(current_vcu_data["g_force_y"])

        # Limit data points
        if len(self.time_data) > MAX_GRAPH_POINTS:
            self.time_data.pop(0)
            self.motor_temp_data.pop(0)
            self.voltage_data.pop(0)
            self.current_data.pop(0)
            self.torque_data.pop(0)
            self.throttle_data.pop(0)
            self.g_force_x_data.pop(0)
            self.g_force_y_data.pop(0)

        # --- Compute Statuses and Trends for KPIs ---
        kpi_display_data = {
            "Lap Time": {"value": lap_time_display, "status": "normal"}, # Lap time value now directly from above
            "SoC": {"value": current_vcu_data["soc"], "unit": "%", "status": compute_status_trend("soc", current_vcu_data["soc"], current_vcu_data["soc_prev"])[0]},
            "Motor Temp": {"value": current_vcu_data["motor_temp"], "unit": "°C", "status": compute_status_trend("motor_temp", current_vcu_data["motor_temp"])[0]},
            "Voltage": {"value": current_vcu_data["voltage"], "unit": "V", "status": compute_status_trend("voltage", current_vcu_data["voltage"])[0]},
            "Current": {"value": current_vcu_data["current"], "unit": "A", "status": compute_status_trend("current", current_vcu_data["current"])[0]},
            "Throttle": {"value": current_vcu_data["throttle"], "unit": "%", "status": compute_status_trend("throttle", current_vcu_data["throttle"])[0]},
            "Torque": {"value": current_vcu_data["torque"], "unit": "Nm", "status": compute_status_trend("torque", current_vcu_data["torque"])[0]},
            "G-Force X": {"value": current_vcu_data["g_force_x"], "unit": "g", "status": compute_status_trend("g_force_x", current_vcu_data["g_force_x"])[0]},
            "G-Force Y": {"value": current_vcu_data["g_force_y"], "unit": "g", "status": compute_status_trend("g_force_y", current_vcu_data["g_force_y"])[0]}
        }

        # Calculate and add Motor Temp Slope KPI
        motor_temp_slope = calculate_slope(self.time_data, self.motor_temp_data, num_points=20)
        motor_temp_slope_status = "normal"
        if motor_temp_slope > 0.5: motor_temp_slope_status = "warning"
        if motor_temp_slope > 1.0: motor_temp_slope_status = "critical"
        kpi_display_data["Motor Temp Slope"] = {"value": f"{motor_temp_slope:.2f}", "unit": "°C/s", "status": motor_temp_slope_status}

        self.update_kpis_display(kpi_display_data)

        # --- Update Graphs ---
        self.update_graphs()

        # --- CSV Logging ---
        self.log_to_csv(current_time_seconds, lap_time_display, current_vcu_data, kpi_display_data)

    def update_time_display(self, current_time_ms):
        current_datetime = datetime.fromtimestamp(current_time_ms / 1000.0)
        self.current_time_label.setText(current_datetime.strftime('%H:%M:%S'))

        session_duration_seconds = (current_time_ms / 1000.0) - self.session_start_time
        hours = int(session_duration_seconds // 3600)
        minutes = int((session_duration_seconds % 3600) // 60)
        seconds = int(session_duration_seconds % 60)
        self.session_duration_label.setText(f"Duration: {hours:02d}:{minutes:02d}:{seconds:02d}")

    def update_kpis_display(self, kpi_data):
        for kpi_name, data in kpi_data.items():
            if kpi_name in self.kpi_labels:
                label = self.kpi_labels[kpi_name]
                display_value = data["value"]
                display_unit = data.get("unit", "")
                status = data.get("status", "normal")

                label.setText(f"{display_value} {display_unit}".strip())
                
                # Apply specific color override if defined in kpi_config
                # Otherwise, apply status colors
                if kpi_name in self.kpi_config and "color_override" in self.kpi_config[kpi_name]:
                    label.setStyleSheet(f"color: {self.kpi_config[kpi_name]['color_override']};")
                elif status == "normal":
                    label.setStyleSheet(f"color: #32CD32;") # LimeGreen
                elif status == "warning":
                    label.setStyleSheet(f"color: #FFA500;") # Orange
                elif status == "critical":
                    label.setStyleSheet(f"color: #FF4500;") # OrangeRed
                else:
                     label.setStyleSheet(f"color: white;") # Default

    def update_graphs(self):
        # pyqtgraph expects x-axis values to be simple numbers.
        # We calculate relative time in seconds from the earliest point in the current data window.
        
        if self.time_data:
            relative_times = [(t - self.time_data[0]) / 1000.0 for t in self.time_data]
        else:
            relative_times = []

        self.temp_curve.setData(relative_times, self.motor_temp_data)
        
        # Voltage/Current Graph
        self.voltage_curve.setData(relative_times, self.voltage_data)
        self.current_curve.setData(relative_times, self.current_data)
        
        # G-Force Graph
        self.g_force_x_curve.setData(relative_times, self.g_force_x_data)
        self.g_force_y_curve.setData(relative_times, self.g_force_y_data)
        
        # Throttle vs. Torque Correlation Graph (REVERTED TO ORIGINAL DUAL-SCATTER DESIGN)
        if len(self.throttle_data) > 0:
            # Historical Data: All points except the very last one
            historical_throttle = self.throttle_data[:-1]
            historical_torque = self.torque_data[:-1]
            self.throttle_torque_historical_scatter.setData(historical_throttle, historical_torque)

            # Current Point: Only the very last point
            current_throttle = self.throttle_data[-1:] # Slicing ensures it's a list for setData
            current_torque = self.torque_data[-1:]
            self.throttle_torque_current_scatter.setData(current_throttle, current_torque)
        else:
            # Clear data if no points
            self.throttle_torque_historical_scatter.setData([])
            self.throttle_torque_current_scatter.setData([])

        # IMPORTANT: No auto-panning for the scatter plot, it maintains its fixed range
        # self.correlation_plot.setXRange(0, 100, padding=0.05) # Already set in initUI
        # self.correlation_plot.setYRange(0, 60, padding=0.05) # Already set in initUI


    def log_to_csv(self, current_time_seconds, lap_time_display, vcu_data, kpi_display_data):
        file_exists = os.path.exists(CSV_FILE)
        with open(CSV_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "Timestamp", "Lap Time", "SoC", "Motor Temp", "Voltage", "Current",
                    "Throttle", "Torque", "G-Force X", "G-Force Y",
                    "Motor Temp Status", "Voltage Status", "Current Status",
                    "Throttle Status", "Torque Status", "G-Force X Status", "G-Force Y Status",
                    "Motor Temp Slope" # Added Slope to CSV
                ])
            
            # For CSV, use the raw numeric values and a single line for lap time
            # Note: lap_time_display for GUI might have newline, remove for CSV if desired
            csv_lap_time = lap_time_display.replace('\n', ' ')

            writer.writerow([
                datetime.fromtimestamp(current_time_seconds).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], # Milliseconds for CSV
                csv_lap_time, vcu_data["soc"], vcu_data["motor_temp"],
                vcu_data["voltage"], vcu_data["current"], vcu_data["throttle"], vcu_data["torque"],
                vcu_data["g_force_x"], vcu_data["g_force_y"],
                kpi_display_data.get("Motor Temp", {}).get("status", "normal"),
                kpi_display_data.get("Voltage", {}).get("status", "normal"),
                kpi_display_data.get("Current", {}).get("status", "normal"),
                kpi_display_data.get("Throttle", {}).get("status", "normal"),
                kpi_display_data.get("Torque", {}).get("status", "normal"),
                kpi_display_data.get("G-Force X", {}).get("status", "normal"),
                kpi_display_data.get("G-Force Y", {}).get("status", "normal"),
                kpi_display_data.get("Motor Temp Slope", {}).get("value", "")
            ])

    def closeEvent(self, event):
        # Ensure the data reading thread is stopped when the window is closed
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Application closing...")
        if self.vcu_reader:
            self.vcu_reader.stop() # Tell worker to stop its loop
        if self.data_thread:
            self.data_thread.quit() # Tell thread to quit
            self.data_thread.wait(3000) # Wait up to 3 seconds for the thread to finish
            if self.data_thread.isRunning():
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Data thread did not terminate gracefully.")
        event.accept()

# --- Main Application Entry Point ---
if __name__ == '__main__':
    # Apply dark theme palette globally
    app = QApplication(sys.argv)
    app.setApplicationDisplayName("EVR Racing Dashboard")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(20, 20, 20)) # Dark background
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(50, 50, 50))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(50, 50, 50))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)

    dashboard = DashboardWindow()
    dashboard.show()
    sys.exit(app.exec_())
