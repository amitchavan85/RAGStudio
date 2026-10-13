import { useMutation, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Eye, FileText, Loader2, Trash2, UploadCloud, XCircle } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api, type Job } from "../api/client";
import { PdfViewer } from "../components/PdfViewer";
import { Badge, Button, cx, IconButton, Modal, Notice } from "../components/ui";
import { friendlyError } from "../lib/logic";
import { useSettings } from "../settings";

export function DocumentsPage() {
  const { settings } = useSettings();
  const queryClient = useQueryClient();
  const options = useQuery({ queryKey: ["options"], queryFn: api.options });
  const docs = useQuery({ queryKey: ["documents", settings.index], queryFn: () => api.documents(settings.index) });
  const [jobIds, setJobIds] = useState<string[]>([]);
  const [uploadError, setUploadError] = useState<unknown>(null);
  const [toDelete, setToDelete] = useState<string | null>(null);
  const [viewing, setViewing] = useState<string | null>(null);
  const strategy = settings.chunking ?? options.data?.defaults.chunking_strategy ?? "recursive";

  const upload = useMutation({
    mutationFn: (files: File[]) => api.upload(settings.index, files, { strategy }),
    onSuccess: (job) => setJobIds((ids) => [job.job_id, ...ids]),
    onError: setUploadError,
  });

  // Poll every running job once a second until it succeeds or fails.
  const jobs = useQueries({
    queries: jobIds.map((id) => ({
      queryKey: ["job", id],
      queryFn: () => api.job(id),
      refetchInterval: (q: { state: { data?: Job } }) =>
        q.state.data && ["succeeded", "failed"].includes(q.state.data.status) ? false : 1000,
    })),
  });
  const finished = jobs.filter((j) => j.data && ["succeeded", "failed"].includes(j.data.status)).length;
  useEffect(() => {
    if (finished) {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      queryClient.invalidateQueries({ queryKey: ["indexes"] });
    }
  }, [finished, queryClient]);

  const del = useMutation({
    mutationFn: (source: string) => api.deleteDocument(settings.index, source),
    onSuccess: () => {
      setToDelete(null);
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      queryClient.invalidateQueries({ queryKey: ["indexes"] });
    },
  });

  const start = (files: File[]) => {
    setUploadError(null);
    const pdfs = files.filter((f) => f.name.toLowerCase().endsWith(".pdf"));
    if (pdfs.length !== files.length) setUploadError(new Error("Only PDF files can be uploaded."));
    if (pdfs.length) upload.mutate(pdfs);
  };

  const list = docs.data?.documents ?? [];
  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-4xl space-y-8 px-6 py-8">
        <header>
          <h1 className="text-2xl font-semibold tracking-tight">Documents</h1>
          <p className="mt-1 text-text-muted">
            Index <span className="font-medium text-text">{settings.index}</span> · split with <span className="font-medium text-text">{strategy}</span> (change in Settings).
            Re-uploading a file with the same name replaces it.
          </p>
        </header>

        <DropZone onFiles={start} busy={upload.isPending} maxMb={options.data?.limits.max_upload_mb} />
        {uploadError != null && <Notice>{uploadError instanceof Error && !("code" in uploadError) ? uploadError.message : friendlyError(uploadError)}</Notice>}

        {jobs.length > 0 && (
          <section aria-label="Uploads" className="space-y-2">
            <h2 className="text-xs font-medium uppercase tracking-wide text-text-muted">Uploads</h2>
            {jobs.map((j, i) => (j.data ? <JobRow key={j.data.job_id} job={j.data} /> : <div key={jobIds[i]} className="skeleton h-14" />))}
          </section>
        )}

        <section aria-label="Documents in this index" className="space-y-2">
          <h2 className="text-xs font-medium uppercase tracking-wide text-text-muted">In this index ({list.length})</h2>
          {docs.isLoading && <div className="space-y-2">{[0, 1, 2].map((i) => <div key={i} className="skeleton h-12" />)}</div>}
          {docs.isError && <Notice>{friendlyError(docs.error)}</Notice>}
          {docs.isSuccess && list.length === 0 && <p className="text-sm text-text-muted">No documents yet. Drop PDFs above to get started.</p>}
          <ul className="divide-y divide-border overflow-hidden rounded-xl border border-border bg-surface">
            {list.map((d) => (
              <li key={d.source} className="flex items-center gap-3 px-4 py-3">
                <FileText size={18} className="shrink-0 text-text-muted" />
                <span className="min-w-0 flex-1 truncate font-medium" title={d.source}>{d.source}</span>
                <Badge>{d.chunks} chunks</Badge>
                <IconButton label={`View ${d.source}`} onClick={() => setViewing(d.source)}><Eye size={16} /></IconButton>
                <IconButton label={`Delete ${d.source}`} onClick={() => setToDelete(d.source)}><Trash2 size={16} /></IconButton>
              </li>
            ))}
          </ul>
        </section>
      </div>

      <Modal open={toDelete != null} onOpenChange={(o) => !o && setToDelete(null)} title="Delete document?" description={`${toDelete} and all its chunks will be removed from this index. This can't be undone.`}>
        {del.isError && <div className="mb-3"><Notice>{friendlyError(del.error)}</Notice></div>}
        <div className="flex justify-end gap-2">
          <Button onClick={() => setToDelete(null)}>Cancel</Button>
          <Button variant="danger" disabled={del.isPending} onClick={() => toDelete && del.mutate(toDelete)}>
            {del.isPending ? <Loader2 size={14} className="animate-spin" /> : <Trash2 size={14} />} Delete
          </Button>
        </div>
      </Modal>

      <Modal open={viewing != null} onOpenChange={(o) => !o && setViewing(null)} title={viewing ?? ""} wide>
        {viewing && <div className="flex h-[70vh] flex-col"><PdfViewer index={settings.index} source={viewing} page={1} /></div>}
      </Modal>
    </div>
  );
}

function DropZone({ onFiles, busy, maxMb }: { onFiles: (f: File[]) => void; busy: boolean; maxMb?: number }) {
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        onFiles([...e.dataTransfer.files]);
      }}
      className={cx(
        "flex flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-10 text-center transition-colors",
        over ? "border-accent bg-tint" : "border-border bg-surface",
      )}
    >
      <UploadCloud size={28} className={over ? "text-tint-text" : "text-text-muted"} />
      <p className="mt-3 font-medium">Drop PDF files here</p>
      <p className="mt-1 text-sm text-text-muted">or choose them from your computer{maxMb ? ` · up to ${maxMb} MB each` : ""}</p>
      <Button variant="primary" className="mt-4" disabled={busy} onClick={() => input.current?.click()}>
        {busy ? <Loader2 size={16} className="animate-spin" /> : <UploadCloud size={16} />} Choose files
      </Button>
      <input
        ref={input}
        type="file"
        accept="application/pdf,.pdf"
        multiple
        hidden
        onChange={(e) => {
          onFiles([...(e.target.files ?? [])]);
          e.target.value = "";
        }}
      />
    </div>
  );
}

function JobRow({ job }: { job: Job }) {
  const running = job.status === "queued" || job.status === "running";
  return (
    <div className="rounded-xl border border-border bg-surface px-4 py-3 animate-fade-in">
      <div className="flex items-center gap-3">
        {running && <Loader2 size={18} className="animate-spin text-accent" />}
        {job.status === "succeeded" && <CheckCircle2 size={18} className="text-success" />}
        {job.status === "failed" && <XCircle size={18} className="text-danger" />}
        <span className="min-w-0 flex-1 truncate text-sm font-medium">{job.files.join(", ")}</span>
        <Badge tone={job.status === "succeeded" ? "success" : job.status === "failed" ? "danger" : "accent"}>
          {job.status === "queued" ? "Queued" : job.status === "running" ? "Processing" : job.status === "succeeded" ? "Done" : "Failed"}
        </Badge>
      </div>
      {job.result && (
        <ul className="mt-2 space-y-0.5 pl-8 text-sm text-text-muted">
          {job.result.files.map((f) => (
            <li key={f.source}>
              {f.source}: {f.pages} pages → {f.chunks} chunks{f.replaced ? ` (replaced ${f.replaced} old chunks)` : ""}
            </li>
          ))}
          {job.result.skipped.map((s) => (
            <li key={s.source} className="text-warning">{s.source}: skipped, {s.reason}</li>
          ))}
        </ul>
      )}
      {job.error && <p className="mt-2 pl-8 text-sm text-danger">{job.error}</p>}
    </div>
  );
}
