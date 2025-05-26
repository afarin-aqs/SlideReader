import { useRef, useState, useEffect } from "react";

const ImageCanvas = ({
  imageSrc,
  circles,
  setCircles,
  clusterMode,
  sendEditCommand = () => {},
}) => {
  const containerRef = useRef(null);
  const [scale, setScale] = useState(1);
  const [translate, setTranslate] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [startPan, setStartPan] = useState({ x: 0, y: 0 });
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      const width = img.width;
      const height = img.height;
      const containerWidth = containerRef.current.offsetWidth;
      const scaleFactor = containerWidth / width;

      setImageSize({ width, height });
      setScale(scaleFactor);
      setTranslate({ x: 0, y: 0 });
    };
    img.src = imageSrc;
  }, [imageSrc]);

  useEffect(() => {
    const ref = containerRef.current;
    const handleWheel = (e) => {
      e.preventDefault();

      const zoomIntensity = 0.001;
      const delta = -e.deltaY;
      const newScale = Math.min(
        Math.max(scale + delta * zoomIntensity, 0.1),
        10,
      );

      const rect = ref.getBoundingClientRect();
      const offsetX = e.clientX - rect.left;
      const offsetY = e.clientY - rect.top;

      const dx = offsetX - translate.x;
      const dy = offsetY - translate.y;

      const newTranslate = {
        x: offsetX - (dx * newScale) / scale,
        y: offsetY - (dy * newScale) / scale,
      };

      setScale(newScale);
      setTranslate(newTranslate);
    };

    ref.addEventListener("wheel", handleWheel, { passive: false });
    return () => ref.removeEventListener("wheel", handleWheel);
  }, [scale, translate]);

  const handleMouseDown = (e) => {
    if (e.target.tagName !== "circle" && e.target.tagName !== "rect") {
      setIsPanning(true);
      setStartPan({ x: e.clientX, y: e.clientY });
    }
  };

  const handleMouseMove = (e) => {
    if (!isPanning) return;
    const dx = e.clientX - startPan.x;
    const dy = e.clientY - startPan.y;
    setTranslate((prev) => ({ x: prev.x + dx, y: prev.y + dy }));
    setStartPan({ x: e.clientX, y: e.clientY });
  };

  const handleMouseUp = () => {
    setIsPanning(false);
  };

  const handleCircleDrag = (e, circle) => {
    e.stopPropagation();
    const startX = e.clientX;
    const startY = e.clientY;

    const initialPositions = {};
    circles.forEach((c) => {
      if (clusterMode && c.cluster === circle.cluster) {
        initialPositions[c.id] = { cx: c.cx, cy: c.cy };
      } else if (!clusterMode && c.id === circle.id) {
        initialPositions[c.id] = { cx: c.cx, cy: c.cy };
      }
    });

    const onMouseMove = (moveEvent) => {
      const dx = (moveEvent.clientX - startX) / scale;
      const dy = (moveEvent.clientY - startY) / scale;

      setCircles((prev) =>
        prev.map((c) =>
          initialPositions[c.id]
            ? {
                ...c,
                cx: initialPositions[c.id].cx + dx,
                cy: initialPositions[c.id].cy + dy,
              }
            : c,
        ),
      );
    };

    const onMouseUp = () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);

      const dx = (window.event.clientX - startX) / scale;
      const dy = (window.event.clientY - startY) / scale;

      sendMoveCommand(circle, dx, dy);
    };

    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
  };

  const handleResizeDrag = (e, circle) => {
    e.stopPropagation();
    const startX = e.clientX;

    const initialRadii = {};
    circles.forEach((c) => {
      if (clusterMode && c.cluster === circle.cluster) {
        initialRadii[c.id] = c.r;
      } else if (!clusterMode && c.id === circle.id) {
        initialRadii[c.id] = c.r;
      }
    });

    const onMouseMove = (moveEvent) => {
      const dx = (moveEvent.clientX - startX) / scale;

      setCircles((prev) =>
        prev.map((c) =>
          initialRadii[c.id]
            ? { ...c, r: Math.max(3, initialRadii[c.id] + dx) }
            : c,
        ),
      );
    };

    const onMouseUp = () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);

      const dx = (window.event.clientX - startX) / scale;
      sendResizeCommand(circle, dx);
    };

    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
  };

  const handleReset = () => {
    const containerWidth = containerRef.current.offsetWidth;
    const scaleFactor = containerWidth / imageSize.width;

    setScale(scaleFactor);
    setTranslate({ x: 0, y: 0 });
  };

  const getClusterColor = (() => {
    const cache = {};
    return (clusterId) => {
      if (clusterId === -1) return "gray";
      if (cache[clusterId]) return cache[clusterId];

      const hue = (clusterId * 137) % 360;
      const color = `hsl(${hue}, 70%, 50%)`;
      cache[clusterId] = color;
      return color;
    };
  })();

  const getCircleIndexInCluster = (circle) => {
    const cluster = circle.cluster;
    const clusterCircles = circles.filter((c) => c.cluster === circle.cluster);
    const sortedClusterCircles = [...clusterCircles].sort(
      (a, b) => a.cx - b.cx,
    );
    const index = sortedClusterCircles.findIndex((c) => c.id === circle.id);
    return index;
  };

  const sendDeleteCommand = (circle) => {
    const cluster = circle.cluster;
    const index = getCircleIndexInCluster(circle);
    sendEditCommand(cluster, `del spot${index}`);
  };

  const sendMoveCommand = (circle, dx, dy) => {
    const cluster = circle.cluster;
    const index = getCircleIndexInCluster(circle);

    const moves = [];
    if (dy !== 0)
      moves.push(`${Math.abs(Math.round(dy))} ${dy > 0 ? "down" : "up"}`);
    if (dx !== 0)
      moves.push(`${Math.abs(Math.round(dx))} ${dx > 0 ? "right" : "left"}`);

    if (moves.length !== 0) {
      moves.forEach((m) => {
        sendEditCommand(cluster, `move spot${index} ${m}`);
      });
    }
  };

  const sendResizeCommand = (circle, dx) => {
    const cluster = circle.cluster;
    const index = getCircleIndexInCluster(circle);
    if (dx !== 0) {
      sendEditCommand(
        cluster,
        `change_r spot${index} r${dx > 0 ? "+" : ""}${Math.round(dx)}`,
      );
    }
  };

  return (
    <div
      ref={containerRef}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      style={{
        width: "100%",
        height: "100%",
        overflow: "hidden",
        position: "relative",
      }}
    >
      <svg
        width={imageSize.width}
        height={imageSize.height}
        style={{
          transform: `translate(${translate.x}px, ${translate.y}px) scale(${scale})`,
          transformOrigin: "0 0",
        }}
      >
        <image
          href={imageSrc}
          x="0"
          y="0"
          width={imageSize.width}
          height={imageSize.height}
        />
        {circles.map((c) => {
          const color = getClusterColor(c.cluster);
          return (
            <g key={c.id}>
              <circle
                cx={c.cx}
                cy={c.cy}
                r={c.r}
                stroke={color}
                strokeWidth="3"
                fill="transparent"
                onMouseDown={(e) => handleCircleDrag(e, c)}
                style={{ cursor: "move" }}
                onMouseEnter={() => {
                  const text = document.getElementById(`cluster-label-${c.id}`);
                  if (text) text.style.display = "block";
                }}
                onMouseLeave={() => {
                  const text = document.getElementById(`cluster-label-${c.id}`);
                  if (text) text.style.display = "none";
                }}
                onDoubleClick={() => {
                  const confirmDelete = window.confirm("Delete this circle?");
                  if (confirmDelete) {
                    sendDeleteCommand(c);
                    setCircles((prev) =>
                      prev.filter((circle) => circle.id !== c.id),
                    );
                  }
                }}
              />
              <rect
                x={c.cx + c.r - 4}
                y={c.cy - 4}
                width={8}
                height={8}
                fill={color}
                style={{ cursor: "ew-resize" }}
                onMouseDown={(e) => handleResizeDrag(e, c)}
              />
              {c.cluster !== -1 && (
                <text
                  id={`cluster-label-${c.id}`}
                  x={c.cx + c.r + 10}
                  y={c.cy}
                  fontSize="24"
                  fill={color}
                  display="none"
                  style={{ display: "none", pointerEvents: "none" }}
                >
                  {c.cluster}
                </text>
              )}
            </g>
          );
        })}
      </svg>

      <button
        onClick={handleReset}
        className="btn btn-outline-secondary"
        style={{ position: "absolute", top: 10, right: 10, zIndex: 10 }}
      >
        Reset View
      </button>
    </div>
  );
};

export default ImageCanvas;
