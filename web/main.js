(() => {
  const streamImg = document.getElementById("stream");
  const offline = document.getElementById("offline");
  const btnAnnotated = document.getElementById("btnAnnotated");
  const btnRaw = document.getElementById("btnRaw");
  const btnStart = document.getElementById("btnStart");
  const btnStop = document.getElementById("btnStop");
  const btnShot = document.getElementById("btnShot");
  const btnFull = document.getElementById("btnFull");
  const openInTab = document.getElementById("openInTab");
  const statusBadges = document.getElementById("statusBadges");
  const sysList = document.getElementById("sysList");
  const nowYear = document.getElementById("year");
  const exportDataBtn = document.getElementById("exportData");
  const clearDataBtn = document.getElementById("clearData");

  let currentType = "annot"; // "annot" | "raw"
  let streaming = false;
  let db = null;
  let fireAlertShown = false;
  let smokeAlertShown = false;

  // Initialize SQLite database
  async function initDatabase() {
    try {
      const SQL = await initSqlJs({
        locateFile: (file) =>
          `https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.8.0/${file}`,
      });

      db = new SQL.Database();

      // Create tables if they don't exist
      db.run(`
        CREATE TABLE IF NOT EXISTS sensor_data (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
          temperature REAL,
          humidity REAL,
          gas_level REAL,
          fire_status REAL,
          ai_confidence REAL,
          alert_triggered BOOLEAN DEFAULT FALSE
        )
      `);

      db.run(`
        CREATE TABLE IF NOT EXISTS detection_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
          event_type TEXT, -- 'fire', 'smoke', 'clear'
          confidence REAL,
          temperature REAL,
          gas_level REAL,
          duration_seconds INTEGER DEFAULT 0
        )
      `);

      console.log("Database initialized successfully");
      updateStats();
    } catch (error) {
      console.error("Failed to initialize database:", error);
    }
  }

  // Store sensor data in database
  function storeSensorData(
    temperature,
    humidity,
    gas,
    fireStatus,
    aiConfidence = null
  ) {
    if (!db) return;

    try {
      db.run(
        `
        INSERT INTO sensor_data (temperature, humidity, gas_level, fire_status, ai_confidence)
        VALUES (?, ?, ?, ?, ?)
      `,
        [temperature, humidity, gas, fireStatus, aiConfidence]
      );

      // Check if we should create a detection event
      const prevStatus = getPreviousFireStatus();
      if (prevStatus !== fireStatus) {
        const eventType =
          fireStatus === 1 ? "fire" : fireStatus === 0.5 ? "smoke" : "clear";
        db.run(
          `
          INSERT INTO detection_events (event_type, confidence, temperature, gas_level)
          VALUES (?, ?, ?, ?)
        `,
          [eventType, aiConfidence || 0.8, temperature, gas]
        );
      }

      updateStats();
    } catch (error) {
      console.error("Error storing sensor data:", error);
    }
  }

  function getPreviousFireStatus() {
    if (!db) return 0;
    try {
      const result = db.exec(`
        SELECT fire_status FROM sensor_data 
        ORDER BY timestamp DESC LIMIT 1
      `);
      return result.length > 0 ? result[0].values[0][0] : 0;
    } catch (error) {
      return 0;
    }
  }

  function updateStats() {
    if (!db) return;

    try {
      // Total fires
      const fireResult = db.exec(`
        SELECT COUNT(*) FROM detection_events WHERE event_type = 'fire'
      `);
      document.getElementById("totalFires").textContent =
        fireResult.length > 0 ? fireResult[0].values[0][0] : 0;

      // Total smoke events
      const smokeResult = db.exec(`
        SELECT COUNT(*) FROM detection_events WHERE event_type = 'smoke'
      `);
      document.getElementById("totalSmoke").textContent =
        smokeResult.length > 0 ? smokeResult[0].values[0][0] : 0;

      // Total clear periods
      const clearResult = db.exec(`
        SELECT COUNT(*) FROM detection_events WHERE event_type = 'clear'
      `);
      document.getElementById("totalClear").textContent =
        clearResult.length > 0 ? clearResult[0].values[0][0] : 0;
    } catch (error) {
      console.error("Error updating stats:", error);
    }
  }

  // Export data as CSV
  function exportData() {
    if (!db) return;

    try {
      const result = db.exec(`
        SELECT timestamp, temperature, humidity, gas_level, fire_status, ai_confidence
        FROM sensor_data ORDER BY timestamp
      `);

      if (result.length === 0) return;

      const headers = [
        "Timestamp",
        "Temperature",
        "Humidity",
        "Gas Level",
        "Fire Status",
        "AI Confidence",
      ];
      const rows = result[0].values;

      let csv = headers.join(",") + "\n";
      rows.forEach((row) => {
        csv += row.join(",") + "\n";
      });

      const blob = new Blob([csv], { type: "text/csv" });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `fire_detection_data_${Date.now()}.csv`;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Error exporting data:", error);
    }
  }

  // Footer year
  nowYear.textContent = new Date().getFullYear();

  // Stream handlers
  function setStreamType(type) {
    currentType = type;
    btnAnnotated.classList.toggle("active", type === "annot");
    btnRaw.classList.toggle("active", type === "raw");
    const url = type === "annot" ? "/video_feed" : "/video_feed_raw";
    openInTab.href = url;
    if (streaming) streamImg.src = url;
  }

  function startStream() {
    streaming = true;
    streamImg.src = currentType === "annot" ? "/video_feed" : "/video_feed_raw";
    btnStart.disabled = true;
    btnStop.disabled = false;
  }

  function stopStream() {
    streaming = false;
    streamImg.src = ""; // stops requests
    btnStart.disabled = false;
    btnStop.disabled = true;
  }

  streamImg.addEventListener("load", () => {
    offline.classList.add("d-none");
  });
  streamImg.addEventListener("error", () => {
    offline.classList.remove("d-none");
  });

  btnAnnotated.addEventListener("click", () => setStreamType("annot"));
  btnRaw.addEventListener("click", () => setStreamType("raw"));
  btnStart.addEventListener("click", startStream);
  btnStop.addEventListener("click", stopStream);

  btnShot.addEventListener("click", () => {
    const a = document.createElement("a");
    a.href = `/snapshot?type=${currentType}`;
    a.download = `snapshot_${currentType}_${Date.now()}.jpg`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  });

  btnFull.addEventListener("click", () => {
    if (streamImg.requestFullscreen) streamImg.requestFullscreen();
  });

  // Status widgets (/status)
  async function loadStatus() {
    try {
      const r = await fetch("/status");
      const s = await r.json();
      statusBadges.innerHTML = `
        <span class="badge text-bg-${
          s.device === "GPU" ? "danger" : "secondary"
        }">
          <i class="fa-solid fa-microchip me-1"></i>${s.device}
        </span>
        <span class="badge text-bg-secondary"><i class="fa-solid fa-maximize me-1"></i>IMG ${
          s.img_max_width
        }px</span>
        <span class="badge text-bg-secondary"><i class="fa-solid fa-forward-step me-1"></i>Skip ${
          s.frame_skip
        }</span>
        <span class="badge text-bg-secondary"><i class="fa-solid fa-film me-1"></i>Stream ${
          s.stream_fps
        } fps</span>
      `;
      sysList.innerHTML = `
        <li><strong>Device:</strong> ${s.device}</li>
        <li><strong>Input resize width:</strong> ${s.img_max_width}</li>
        <li><strong>Frame skip:</strong> ${s.frame_skip}</li>
        <li><strong>Stream FPS:</strong> ${s.stream_fps}</li>
        <li><strong>Target inference FPS:</strong> ${s.target_inf_fps}</li>
      `;
    } catch (e) {
      // ignore
    }
  }

  // Calculate fire status based on sensor values
  function calculateFireStatus(temperature, gas, humidity) {
    // Simple heuristic for fire detection
    // You can modify these thresholds based on your requirements

    let status = 0; // No fire

    // High temperature and high gas = fire
    if (temperature > 50 && gas > 2000) {
      status = 1; // Fire
    }
    // Moderate gas levels = smoke
    else if (gas > 1000000) {
      status = 0.5; // Smoke
    }
    // High temperature alone might indicate potential fire
    else if (temperature > 60) {
      status = 0.5; // Smoke/Potential fire
    }

    return status;
  }

  // Show alert modals based on fire status
  function handleFireAlerts(fireStatus, temperature, gas) {
    const fireStatusBadge = document.getElementById("currentFireStatus");
    const fireValueBadge = document.getElementById("currentFireValue");

    // Update status badges
    fireValueBadge.textContent = fireStatus;
    fireStatusBadge.className = "badge";

    if (fireStatus === 1) {
      fireStatusBadge.classList.add("fire-status-1");
      fireStatusBadge.textContent = "FIRE DETECTED";

      // Show fire alert modal
      if (!fireAlertShown) {
        document.getElementById("alertTemp").textContent =
          temperature.toFixed(1);
        document.getElementById("alertGas").textContent = gas.toFixed(0);
        const fireModal = new bootstrap.Modal(
          document.getElementById("fireAlertModal")
        );
        fireModal.show();
        fireAlertShown = true;
        smokeAlertShown = false;
      }
    } else if (fireStatus === 0.5) {
      fireStatusBadge.classList.add("fire-status-0_5");
      fireStatusBadge.textContent = "SMOKE DETECTED";

      // Show smoke alert modal
      if (!smokeAlertShown) {
        document.getElementById("alertSmokeGas").textContent = gas.toFixed(0);
        const smokeModal = new bootstrap.Modal(
          document.getElementById("smokeAlertModal")
        );
        smokeModal.show();
        smokeAlertShown = true;
        fireAlertShown = false;
      }
    } else {
      fireStatusBadge.classList.add("fire-status-0");
      fireStatusBadge.textContent = "NO FIRE";
      fireAlertShown = false;
      smokeAlertShown = false;
    }
  }

  // Charts
  function createChart(ctx, label, color) {
    return new Chart(ctx, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          { label, data: [], borderColor: color, tension: 0.25, fill: false },
        ],
      },
      options: {
        animation: false,
        responsive: true,
        scales: {
          x: { ticks: { color: "#888", maxTicksLimit: 8 } },
          y: { ticks: { color: "#888" } },
        },
        plugins: { legend: { labels: { color: "#aaa" } } },
      },
    });
  }

  // Binary chart for fire detection
  function createFireChart(ctx) {
    return new Chart(ctx, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Fire Status",
            data: [],
            borderColor: "#e74c3c",
            backgroundColor: "rgba(231, 76, 60, 0.1)",
            tension: 0,
            fill: true,
            pointBackgroundColor: function (context) {
              const value = context.dataset.data[context.dataIndex];
              return value === 1
                ? "#e74c3c"
                : value === 0.5
                ? "#f39c12"
                : "#27ae60";
            },
          },
        ],
      },
      options: {
        animation: false,
        responsive: true,
        scales: {
          x: {
            ticks: { color: "#888", maxTicksLimit: 8 },
          },
          y: {
            ticks: {
              color: "#888",
              callback: function (value) {
                if (value === 0) return "No Fire";
                if (value === 0.5) return "Smoke";
                if (value === 1) return "Fire";
                return value;
              },
            },
            min: 0,
            max: 1,
          },
        },
        plugins: {
          legend: {
            labels: { color: "#aaa" },
          },
          tooltip: {
            callbacks: {
              label: function (context) {
                const value = context.raw;
                if (value === 0) return "Status: No Fire (0)";
                if (value === 0.5) return "Status: Smoke (0.5)";
                if (value === 1) return "Status: Fire (1)";
                return `Status: ${value}`;
              },
            },
          },
        },
      },
    });
  }

  const c1 = createChart(
    document.getElementById("chart1"),
    "Field 1",
    "#e74c3c"
  );
  const c2 = createChart(
    document.getElementById("chart2"),
    "Field 2",
    "#3498db"
  );
  const c3 = createChart(
    document.getElementById("chart3"),
    "Field 3",
    "#2ecc71"
  );
  const fireChart = createFireChart(document.getElementById("fireChart"));

  function pushPoint(chart, label, value) {
    chart.data.labels.push(label);
    chart.data.datasets[0].data.push(value);
    if (chart.data.labels.length > 20) {
      chart.data.labels.shift();
      chart.data.datasets[0].data.shift();
    }
    chart.update();
  }

  async function pollSensors() {
    try {
      const r = await fetch("/sensor_data");
      const d = await r.json();
      const now = new Date().toLocaleTimeString();

      const temperature = d.field1?.toFixed?.(2) ?? d.field1 ?? 0;
      const humidity = d.field2?.toFixed?.(2) ?? d.field2 ?? 0;
      const gas = d.field3?.toFixed?.(2) ?? d.field3 ?? 0;

      document.getElementById("m1").textContent = temperature;
      document.getElementById("m2").textContent = humidity;
      document.getElementById("m3").textContent = gas;
      document.getElementById(
        "lastUpdated"
      ).textContent = `Last update: ${now}`;

      // Calculate fire status
      const fireStatus = calculateFireStatus(
        parseFloat(temperature),
        parseFloat(gas),
        parseFloat(humidity)
      );

      // Handle alerts
      handleFireAlerts(fireStatus, parseFloat(temperature), parseFloat(gas));

      // Store data in database
      storeSensorData(
        parseFloat(temperature),
        parseFloat(humidity),
        parseFloat(gas),
        fireStatus
      );

      // Update charts
      pushPoint(c1, now, temperature || 0);
      pushPoint(c2, now, humidity || 0);
      pushPoint(c3, now, gas || 0);
      pushPoint(fireChart, now, fireStatus);
    } catch (e) {
      console.error("Error polling sensors:", e);
    }
  }

  // Event listeners for database operations
  exportDataBtn.addEventListener("click", exportData);
  clearDataBtn.addEventListener("click", () => {
    if (
      confirm("Are you sure you want to clear all data? This cannot be undone.")
    ) {
      if (db) {
        db.run("DELETE FROM sensor_data");
        db.run("DELETE FROM detection_events");
        updateStats();
        alert("All data cleared successfully.");
      }
    }
  });

  // Initialize and start
  initDatabase();
  loadStatus();
  setStreamType("annot");
  startStream();
  pollSensors();
  setInterval(pollSensors, 5000);
})();
