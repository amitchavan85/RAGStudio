import { KeyRound } from "lucide-react";
import { useEffect, useState } from "react";
import { getApiKey, setApiKey } from "../api/client";
import { Button, Modal } from "./ui";

/** Opens automatically when the server answers 401; the key is stored in this browser only. */
export function ApiKeyDialog({ open, onOpenChange, onSaved }: { open: boolean; onOpenChange: (o: boolean) => void; onSaved: () => void }) {
  const [value, setValue] = useState("");
  useEffect(() => {
    if (open) setValue(getApiKey());
  }, [open]);

  const save = (key: string) => {
    setApiKey(key.trim());
    onOpenChange(false);
    onSaved();
  };

  return (
    <Modal open={open} onOpenChange={onOpenChange} title="API key" description="This server requires a key. It is stored in this browser only and sent as the X-API-Key header.">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          save(value);
        }}
        className="space-y-4"
      >
        <label className="block">
          <span className="sr-only">API key</span>
          <div className="flex items-center gap-2 rounded-lg border border-border bg-bg px-3 focus-within:border-accent">
            <KeyRound size={16} className="text-text-muted" aria-hidden />
            <input
              type="password"
              autoComplete="off"
              autoFocus
              value={value}
              onChange={(e) => setValue(e.target.value)}
              placeholder="Paste your API key"
              className="h-10 w-full bg-transparent text-sm outline-none"
            />
          </div>
        </label>
        <div className="flex justify-between">
          <Button type="button" variant="ghost" onClick={() => save("")}>Remove key</Button>
          <Button type="submit" variant="primary" disabled={!value.trim()}>Save</Button>
        </div>
      </form>
    </Modal>
  );
}
