document.addEventListener("DOMContentLoaded", () => {


    const canvas = document.getElementById("activityChart");
    const rawDates = Array.isArray(window.matrixmatchResearcherDates)
        ? window.matrixmatchResearcherDates
        : [];
    if (!canvas || !rawDates.length) {
        return;
    }

    const context = canvas.getContext("2d");
    if (!context) {
        return;
    }

    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const dayKeys = [];
    for (let offset = 6; offset >= 0; offset -= 1) {
        const date = new Date(today);
        date.setDate(today.getDate() - offset);
        dayKeys.push(date.toISOString().slice(0, 10));
    }

    const countMap = Object.fromEntries(dayKeys.map((key) => [key, 0]));
    rawDates.forEach((value) => {
        if (!value) {
            return;
        }
        if (countMap[value] !== undefined) {
            countMap[value] += 1;
        }
    });

    const values = dayKeys.map((key) => countMap[key]);
    const width = canvas.width;
    const height = canvas.height;
    const leftPadding = 44;
    const rightPadding = 20;
    const topPadding = 20;
    const bottomPadding = 38;
    const chartWidth = width - leftPadding - rightPadding;
    const chartHeight = height - topPadding - bottomPadding;
    const maxValue = Math.max(...values, 1);

    context.clearRect(0, 0, width, height);

    context.strokeStyle = "rgba(130, 149, 176, 0.28)";
    context.lineWidth = 1;
    for (let i = 0; i <= 4; i += 1) {
        const y = topPadding + (chartHeight / 4) * i;
        context.beginPath();
        context.moveTo(leftPadding, y);
        context.lineTo(width - rightPadding, y);
        context.stroke();
    }

    const points = values.map((value, index) => {
        const x = leftPadding + (chartWidth / (values.length - 1)) * index;
        const y = topPadding + chartHeight - (value / maxValue) * chartHeight;
        return { x, y, value };
    });

    const fillGradient = context.createLinearGradient(0, topPadding, 0, topPadding + chartHeight);
    fillGradient.addColorStop(0, "rgba(30, 166, 246, 0.34)");
    fillGradient.addColorStop(1, "rgba(30, 166, 246, 0.03)");

    context.beginPath();
    context.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i += 1) {
        const prev = points[i - 1];
        const point = points[i];
        const controlX = (prev.x + point.x) / 2;
        context.quadraticCurveTo(controlX, prev.y, point.x, point.y);
    }
    context.lineTo(points[points.length - 1].x, topPadding + chartHeight);
    context.lineTo(points[0].x, topPadding + chartHeight);
    context.closePath();
    context.fillStyle = fillGradient;
    context.fill();

    context.beginPath();
    context.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i += 1) {
        const prev = points[i - 1];
        const point = points[i];
        const controlX = (prev.x + point.x) / 2;
        context.quadraticCurveTo(controlX, prev.y, point.x, point.y);
    }
    context.strokeStyle = "#1ea6f6";
    context.lineWidth = 3;
    context.stroke();

    context.fillStyle = "#1ea6f6";
    points.forEach((point) => {
        context.beginPath();
        context.arc(point.x, point.y, 4, 0, Math.PI * 2);
        context.fill();
    });

    context.fillStyle = "#7f8ca0";
    context.font = "12px Plus Jakarta Sans, sans-serif";
    dayKeys.forEach((key, index) => {
        const x = leftPadding + (chartWidth / (dayKeys.length - 1)) * index;
        const label = key.slice(5);
        context.fillText(label, x - 14, height - 14);
    });
});
