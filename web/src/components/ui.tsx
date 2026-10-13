/** Small shared building blocks styled with the theme tokens. */
import * as Dialog from "@radix-ui/react-dialog";
import * as Tooltip from "@radix-ui/react-tooltip";
import { X } from "lucide-react";
import type { ButtonHTMLAttributes, ReactNode } from "react";

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "ghost" | "outline" | "danger"; size?: "sm" | "md" };

export function Button({ variant = "outline", size = "md", className, ...props }: ButtonProps) {
  return (
    <button
      {...props}
      className={cx(
        "inline-flex items-center justify-center gap-2 rounded-lg font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50",
        size === "sm" ? "h-8 px-3 text-sm" : "h-10 px-4 text-sm",
        variant === "primary" && "bg-accent text-on-accent hover:opacity-90",
        variant === "outline" && "border border-border bg-surface text-text hover:bg-surface-2",
        variant === "ghost" && "text-text-muted hover:bg-surface-2 hover:text-text",
        variant === "danger" && "bg-danger text-white hover:opacity-90",
        className,
      )}
    />
  );
}

export function IconButton({ label, children, ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { label: string }) {
  return (
    <Tip text={label}>
      <button
        aria-label={label}
        {...props}
        className={cx("inline-flex h-9 w-9 items-center justify-center rounded-lg text-text-muted transition-colors hover:bg-surface-2 hover:text-text", props.className)}
      >
        {children}
      </button>
    </Tip>
  );
}

export function Tip({ text, children }: { text: ReactNode; children: ReactNode }) {
  return (
    <Tooltip.Root delayDuration={300}>
      <Tooltip.Trigger asChild>{children}</Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content sideOffset={6} className="z-50 max-w-xs rounded-md border border-border bg-surface px-2.5 py-1.5 text-xs text-text shadow-lg animate-fade-in">
          {text}
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}

export function Modal({ open, onOpenChange, title, description, children, wide }: {
  open: boolean; onOpenChange: (o: boolean) => void; title: string; description?: string; children: ReactNode; wide?: boolean;
}) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/40 backdrop-blur-[2px] animate-fade-in" />
        <Dialog.Content
          className={cx(
            "fixed left-1/2 top-1/2 z-50 max-h-[85vh] w-[92vw] -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-xl border border-border bg-surface p-6 shadow-2xl animate-fade-in",
            wide ? "max-w-2xl" : "max-w-md",
          )}
        >
          <div className="mb-4 flex items-start justify-between gap-4">
            <div>
              <Dialog.Title className="text-lg font-semibold">{title}</Dialog.Title>
              {description && <Dialog.Description className="mt-1 text-sm text-text-muted">{description}</Dialog.Description>}
            </div>
            <Dialog.Close asChild>
              <button aria-label="Close" className="rounded-md p-1 text-text-muted hover:bg-surface-2 hover:text-text">
                <X size={18} />
              </button>
            </Dialog.Close>
          </div>
          {children}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

export function Badge({ tone = "neutral", children }: { tone?: "neutral" | "success" | "warning" | "danger" | "accent"; children: ReactNode }) {
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium",
        tone === "neutral" && "bg-surface-2 text-text-muted",
        tone === "success" && "bg-surface-2 text-success",
        tone === "warning" && "bg-surface-2 text-warning",
        tone === "danger" && "bg-surface-2 text-danger",
        tone === "accent" && "bg-tint text-tint-text",
      )}
    >
      {children}
    </span>
  );
}

export function Notice({ tone = "danger", children }: { tone?: "danger" | "warning" | "info"; children: ReactNode }) {
  return (
    <div
      role={tone === "danger" ? "alert" : "status"}
      className={cx(
        "rounded-lg border px-3 py-2 text-sm",
        tone === "danger" && "border-danger/40 text-danger",
        tone === "warning" && "border-warning/40 text-warning",
        tone === "info" && "border-border text-text-muted",
      )}
    >
      {children}
    </div>
  );
}

export function Label({ children, hint }: { children: ReactNode; hint?: string }) {
  return (
    <div className="mb-1.5">
      <div className="text-sm font-medium">{children}</div>
      {hint && <div className="text-xs text-text-muted">{hint}</div>}
    </div>
  );
}

export function Segmented<T extends string>({ value, options, onChange, ariaLabel }: {
  value: T; options: { id: T; label: string; title?: string; disabled?: boolean }[]; onChange: (v: T) => void; ariaLabel: string;
}) {
  return (
    <div role="radiogroup" aria-label={ariaLabel} className="inline-flex rounded-lg border border-border bg-surface-2 p-0.5">
      {options.map((o) => (
        <button
          key={o.id}
          role="radio"
          aria-checked={value === o.id}
          title={o.title}
          disabled={o.disabled}
          onClick={() => onChange(o.id)}
          className={cx(
            "rounded-md px-3 py-1 text-sm transition-colors disabled:cursor-not-allowed disabled:opacity-40",
            value === o.id ? "bg-surface text-text shadow-sm" : "text-text-muted hover:text-text",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
