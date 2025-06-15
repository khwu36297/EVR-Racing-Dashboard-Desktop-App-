# EVR-Racing-Dashboard-Desktop-App

# EVR Racing Dashboard (PyQt5, PySerial, PyQtGraph)

This project provides a real-time monitoring dashboard for Electric Vehicle Racing (EVR) data, built using PyQt5 for the graphical user interface, PySerial for communication with a Vehicle Control Unit (VCU), and PyQtGraph for dynamic data visualization.

## Features

* **Real-time Data Acquisition**: Connects to a VCU (or simulates data) to fetch live performance metrics.
* **Dynamic KPI Display**: Shows crucial performance indicators like:
    * **Lap Time**: Tracks current and best lap times.
    * **State of Charge (SoC)**: Battery percentage with color-coded status (normal, warning, critical).
    * **Motor Temperature**: Real-time temperature with status alerts and **calculated rate of change (slope)**.
    * **Voltage & Current**: Electrical system health.
    * **Throttle & Torque**: Motor performance metrics.
    * **G-Forces (X & Y)**: Real-time acceleration/deceleration forces.
* **Interactive Time-Series Graphs**: Visualizes trends for Motor Temperature, Voltage, Current, and G-Forces over time.
* **Throttle vs. Torque Correlation Plot**: A scatter plot to analyze the relationship between throttle input and motor torque output, with historical and current data points.
* **Color-Coded Status & Trend Indicators**: KPIs dynamically change color (e.g., green for normal, orange for warning, red for critical) based on predefined thresholds.
* **Multi-threaded Architecture**: Data acquisition runs in a separate thread to ensure a responsive UI.
* **CSV Logging**: Automatically logs all incoming data to a CSV file for post-analysis.
* **Mock Data Mode**: Easily switch between live serial data and simulated data for development and testing without a physical VCU.

## Technologies Used

* **PyQt5**: For building the rich desktop graphical user interface.
* **PySerial**: For serial communication with the VCU.
* **PyQtGraph**: For high-performance, interactive plotting and data visualization.
* **NumPy**: For numerical operations, specifically for calculating data slopes.
* **Python's `csv` and `datetime` modules**: For data logging and timestamping.

## Getting Started

### Prerequisites

Before running the application, ensure you have Python 3.x installed. Then, install the required libraries:

```bash
pip install PyQt5 pyserial pyqtgraph numpy
```

### Running the Application

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/your-repo-name.git
    cd your-repo-name
    ```

2.  **Configure Data Source:**
    Open the `main.py` (or whatever your main file is named) and adjust the `USE_MOCK_DATA` global variable:
    ```python
    # --- Global Configuration ---
    # Set to True for mock data, False for actual Serial Port
    USE_MOCK_DATA = True # Set to False to use actual serial port
    ```

    * **For Mock Data (Default)**: No further configuration is needed. The dashboard will generate simulated data.
    * **For Actual Serial Port**:
        Set `USE_MOCK_DATA = False`.
        Adjust `SERIAL_PORT` and `BAUD_RATE` to match your VCU's serial configuration:
        ```python
        SERIAL_PORT = 'COM4' # e.g., 'COM1' on Windows, '/dev/ttyUSB0' on Linux, '/dev/pts/1' for virtual serial port
        BAUD_RATE = 115200
        ```
        Ensure your VCU is sending data in the expected comma-separated format: `motor_temp,voltage,current,throttle,torque,g_force_x,g_force_y,soc`.

3.  **Run the Dashboard:**
    ```bash
    python your_main_dashboard_file.py
    ```

## Code Structure Highlights

* **`VcuDataReader(QObject)`**: A `QObject` subclass running in a separate `QThread` to handle data acquisition (either mock or serial) without blocking the UI. It emits a `data_received` signal when new data is available.
* **`DashboardWindow(QMainWindow)`**: The main application window responsible for setting up the UI, connecting signals/slots, updating KPI displays, and plotting data.
* **`generate_mock_data()`**: Simulates VCU data for testing purposes.
* **`compute_status_trend()`**: Determines the visual status (normal, warning, critical) and trend (up, down, steady) for each KPI.
* **`calculate_slope()`**: Uses NumPy to calculate the rate of change for metrics like motor temperature, providing predictive insights.
* **`_create_graph_widget()`**: A helper method to streamline the creation and configuration of PyQtGraph plots, including zoom reset functionality and curve visibility toggles via checkboxes.

## CSV Logging

The application automatically logs all received data to `evr_data_log.csv` in the same directory as the script. The file is cleared each time the application starts to ensure a fresh log for each session.

## Screenshots (Optional, but highly recommended)

*Include screenshots of your dashboard here to give users a visual idea of what it looks like.*

## Contact

For any questions or suggestions, please contact the developer: sorasaklaopraphaiphan@gmail.com

---
