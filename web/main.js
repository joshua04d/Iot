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

  let currentType = "annot"; // "annot" | "raw"
  let streaming = false;

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

  // Charts
  function createChart(ctx, label, color) {
    return new Chart(ctx, {
      type: "line",
      data: {
        labels: [],
        datasets: [{ label, data: [], borderColor: color, tension: 0.25 }],
      },
      options: {
        animation: false,
        responsive: true,
        scales: {
          x: { ticks: { color: "#888" } },
          y: { ticks: { color: "#888" } },
        },
        plugins: { legend: { labels: { color: "#aaa" } } },
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

  function pushPoint(chart, label, value) {
    chart.data.labels.push(label);
    chart.data.datasets[0].data.push(value);
    if (chart.data.labels.length > 30) {
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
      document.getElementById("m1").textContent =
        d.field1?.toFixed?.(2) ?? d.field1 ?? "--";
      document.getElementById("m2").textContent =
        d.field2?.toFixed?.(2) ?? d.field2 ?? "--";
      document.getElementById("m3").textContent =
        d.field3?.toFixed?.(2) ?? d.field3 ?? "--";
      document.getElementById(
        "lastUpdated"
      ).textContent = `Last update: ${now}`;
      pushPoint(c1, now, d.field1 || 0);
      pushPoint(c2, now, d.field2 || 0);
      pushPoint(c3, now, d.field3 || 0);
    } catch (e) {
      // soft-fail
    }
  }

  // Init
  loadStatus();
  setStreamType("annot");
  startStream();
  pollSensors();
  setInterval(pollSensors, 5000);
})();
