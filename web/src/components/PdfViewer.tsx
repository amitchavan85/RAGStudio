import { useQuery } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, Highlighter, ZoomIn, ZoomOut } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import { friendlyError, matchItems } from "../lib/logic";
import { IconButton, Notice } from "./ui";

type PdfDoc = { numPages: number; getPage: (n: number) => Promise<any> };
type Rect = { left: number; top: number; width: number; height: number };
type Matrix = [number, number, number, number, number, number];

/** Loads pdf.js only when a document is first opened, keeping the initial bundle small. */
async function openPdf(blob: Blob): Promise<PdfDoc> {
  const pdfjs = await import("pdfjs-dist");
  pdfjs.GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url).toString();
  return pdfjs.getDocument({ data: await blob.arrayBuffer() }).promise;
}

/** Product of two PDF transformation matrices [a, b, c, d, e, f]. */
function multiply(m: Matrix, n: Matrix): Matrix {
  return [
    m[0] * n[0] + m[2] * n[1], m[1] * n[0] + m[3] * n[1],
    m[0] * n[2] + m[2] * n[3], m[1] * n[2] + m[3] * n[3],
    m[0] * n[4] + m[2] * n[5] + m[4], m[1] * n[4] + m[3] * n[5] + m[5],
  ];
}

/** Boxes (CSS pixels) around the page's text items that belong to the passage. */
async function passageRects(page: any, viewport: { transform: Matrix; scale: number }, passage: string): Promise<Rect[]> {
  const content = await page.getTextContent();
  const items = content.items.filter((it: any) => typeof it.str === "string");
  const hits = matchItems(items.map((it: any) => it.str), passage);
  const rects: Rect[] = [];
  for (const i of [...hits].sort((a, b) => a - b)) {
    const it = items[i];
    if (!it.str.trim()) continue;
    const t = multiply(viewport.transform, it.transform as Matrix);
    const fontHeight = Math.hypot(t[2], t[3]);
    rects.push({ left: t[4] - 1, top: t[5] - fontHeight - 1, width: it.width * viewport.scale + 2, height: fontHeight * 1.2 + 2 });
  }
  return rects;
}

export function PdfViewer({ index, source, page, highlight }: {
  index: string; source: string; page: number; highlight?: string; // highlight: passage to mark on `page`
}) {
  const file = useQuery({ queryKey: ["file", index, source], queryFn: () => api.documentFile(index, source), staleTime: Infinity });
  const [doc, setDoc] = useState<PdfDoc | null>(null);
  const [current, setCurrent] = useState(page);
  const [zoom, setZoom] = useState(1);
  const [size, setSize] = useState<{ width: number; height: number } | null>(null);
  const [rects, setRects] = useState<Rect[]>([]);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const boxRef = useRef<HTMLDivElement>(null);
  const firstMark = useRef<HTMLDivElement>(null);

  useEffect(() => setCurrent(page), [page, source]);

  useEffect(() => {
    let cancelled = false;
    setDoc(null);
    if (file.data) openPdf(file.data).then((d) => !cancelled && setDoc(d));
    return () => {
      cancelled = true;
    };
  }, [file.data]);

  useEffect(() => {
    if (!doc || !canvasRef.current || !boxRef.current) return;
    let task: { cancel: () => void; promise: Promise<void> } | null = null;
    let cancelled = false;
    setRects([]);
    doc.getPage(Math.min(Math.max(1, current), doc.numPages)).then(async (p) => {
      if (cancelled || !canvasRef.current || !boxRef.current) return;
      const fit = (boxRef.current.clientWidth - 2) / p.getViewport({ scale: 1 }).width;
      const ratio = window.devicePixelRatio || 1;
      const css = p.getViewport({ scale: fit * zoom }); // layout size (CSS pixels)
      const viewport = p.getViewport({ scale: fit * zoom * ratio }); // canvas size (device pixels: sharp on HiDPI)
      const canvas = canvasRef.current;
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      setSize({ width: css.width, height: css.height });
      task = p.render({ canvas, viewport });
      task!.promise.catch(() => undefined);
      if (highlight && current === page) {
        const found = await passageRects(p, css, highlight);
        if (!cancelled) setRects(found);
      }
    });
    return () => {
      cancelled = true;
      task?.cancel();
    };
  }, [doc, current, zoom, highlight, page]);

  // Bring the passage into view once it is marked.
  useEffect(() => {
    firstMark.current?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [rects]);

  if (file.isError) return <Notice tone="info">{friendlyError(file.error)}</Notice>;

  const onCitedPage = highlight && current === page;
  return (
    <div className="flex min-h-0 flex-col gap-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1">
          <IconButton label="Previous page" disabled={current <= 1} onClick={() => setCurrent((c) => c - 1)}>
            <ChevronLeft size={16} />
          </IconButton>
          <span className="min-w-20 text-center text-sm text-text-muted" aria-live="polite">
            {doc ? `Page ${current} of ${doc.numPages}` : "Loading…"}
          </span>
          <IconButton label="Next page" disabled={!doc || current >= doc.numPages} onClick={() => setCurrent((c) => c + 1)}>
            <ChevronRight size={16} />
          </IconButton>
          {onCitedPage && doc && (
            <span className="ml-1 inline-flex items-center gap-1 text-xs text-text-muted" aria-live="polite">
              <Highlighter size={13} /> {rects.length ? "Cited passage highlighted" : "Passage not located on this page"}
            </span>
          )}
          {highlight && !onCitedPage && doc && (
            <button className="ml-1 text-xs text-accent hover:underline" onClick={() => setCurrent(page)}>Back to cited page {page}</button>
          )}
        </div>
        <div className="flex items-center gap-1">
          <IconButton label="Zoom out" disabled={zoom <= 0.6} onClick={() => setZoom((z) => +(z - 0.2).toFixed(1))}>
            <ZoomOut size={16} />
          </IconButton>
          <span className="w-10 text-center text-xs text-text-muted">{Math.round(zoom * 100)}%</span>
          <IconButton label="Zoom in" disabled={zoom >= 2.4} onClick={() => setZoom((z) => +(z + 0.2).toFixed(1))}>
            <ZoomIn size={16} />
          </IconButton>
        </div>
      </div>
      <div ref={boxRef} className="min-h-0 flex-1 overflow-auto rounded-lg border border-border bg-surface-2">
        {!doc && <div className="skeleton m-3 h-96" />}
        <div className={doc ? "relative mx-auto" : "hidden"} style={size ? { width: size.width, height: size.height } : undefined}>
          <canvas ref={canvasRef} className="block bg-white" style={size ? { width: size.width, height: size.height } : undefined}
            aria-label={`${source}, page ${current}`} />
          {rects.map((r, i) => (
            <div
              key={i}
              ref={i === 0 ? firstMark : undefined}
              data-testid="passage-mark"
              aria-hidden
              className="pointer-events-none absolute rounded-[3px] mix-blend-multiply"
              style={{ left: r.left, top: r.top, width: r.width, height: r.height, background: "rgba(250, 204, 21, 0.45)" }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
