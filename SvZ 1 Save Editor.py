#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import struct
import hashlib
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from typing import Optional

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad


# ----------------------------
# Crypto (matches Unity code)
# ----------------------------

PASSWORD = "_Glu#Games$2012_"
HASH_NAME = "sha256"
ITERATIONS = 1000


def password_derive_bytes(password: str, salt: bytes, hash_name: str, iterations: int, cb: int) -> bytes:
    if iterations <= 0:
        raise ValueError("iterations must be > 0")

    pwd_bytes = password.encode("utf-8")
    h = hashlib.new(hash_name)
    h.update(pwd_bytes)
    h.update(salt or b"")
    base_value = h.digest()

    for _ in range(1, iterations - 1):
        base_value = hashlib.new(hash_name, base_value).digest()

    out = bytearray()
    prefix = 0
    while len(out) < cb:
        data = b""
        if prefix > 0:
            data = str(prefix).encode("ascii")
        prefix += 1

        blk = hashlib.new(hash_name)
        blk.update(data)
        blk.update(base_value)
        out.extend(blk.digest())

        if prefix > 999 and len(out) < cb:
            raise ValueError("Too many bytes requested (PasswordDeriveBytes prefix limit)")
    return bytes(out[:cb])


def derive_key(password: str, salt: bytes) -> bytes:
    return password_derive_bytes(password, salt, HASH_NAME, ITERATIONS, 32)


def decrypt_save_file(path_in: str) -> bytes:
    with open(path_in, "rb") as f:
        header = f.read(32)
        if len(header) != 32:
            raise ValueError("File too short (missing IV+salt).")
        iv = header[:16]
        salt = header[16:32]
        ciphertext = f.read()

    key = derive_key(PASSWORD, salt)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded_plain = cipher.decrypt(ciphertext)

    try:
        plain = unpad(padded_plain, 16, style="pkcs7")
    except ValueError as e:
        raise ValueError("Bad padding (wrong password/format?)") from e

    if len(plain) < 8 + 32:
        raise ValueError("Decrypted payload too short.")

    data_len = struct.unpack("<q", plain[:8])[0]
    if data_len < 0:
        raise ValueError(f"Invalid data length: {data_len}")

    expected_total = 8 + data_len + 32
    if len(plain) != expected_total:
        raise ValueError(f"Payload size mismatch. Got {len(plain)} bytes, expected {expected_total}.")

    data = plain[8:8 + data_len]
    file_hash = plain[8 + data_len:8 + data_len + 32]
    calc_hash = hashlib.sha256(data).digest()

    if file_hash != calc_hash:
        raise ValueError("SHA256 mismatch (corrupted save or wrong key).")

    return data


def encrypt_save_file(data: bytes, path_out: str) -> None:
    iv = os.urandom(16)
    salt = os.urandom(16)
    key = derive_key(PASSWORD, salt)

    payload = struct.pack("<q", len(data)) + data + hashlib.sha256(data).digest()
    padded = pad(payload, 16, style="pkcs7")

    cipher = AES.new(key, AES.MODE_CBC, iv)
    ciphertext = cipher.encrypt(padded)

    with open(path_out, "wb") as f:
        f.write(iv)
        f.write(salt)
        f.write(ciphertext)


# ----------------------------
# Parsing / serialization
# ----------------------------

SECTION_RE = re.compile(r'^\s*\[(.+?)\]\s*$')
KV_RE = re.compile(r'^(\s*)([^=\r\n]+?)\s*=\s*(.*)$')


class Entry:
    def __init__(self, section, indent, key, value, order):
        self.section = section
        self.indent = indent
        self.key = key
        self.value = value
        self.order = order
        self._orig_key = key
        self._orig_value = value

    @property
    def changed(self):
        return (self.key != self._orig_key) or (self.value != self._orig_value)


def parse_save_text(text: str):
    section = None
    entries = []
    order = 0
    current_entry = None

    for raw_line in text.splitlines():
        msec = SECTION_RE.match(raw_line)
        if msec:
            section = msec.group(1).strip()
            current_entry = None
            continue

        mkv = KV_RE.match(raw_line)
        if mkv:
            indent, key, value = mkv.group(1), mkv.group(2).strip(), mkv.group(3)
            current_entry = Entry(section, indent, key, value, order)
            entries.append(current_entry)
            order += 1
            continue

        if raw_line.strip() == "":
            current_entry = None
            continue

        if current_entry is not None:
            current_entry.value += "\n" + raw_line

    return entries


def entry_to_lines(e: Entry):
    if "\n" not in e.value:
        return [f"{e.indent}{e.key} = {e.value}"]
    parts = e.value.split("\n")
    lines = [f"{e.indent}{e.key} = {parts[0]}"]
    lines.extend(parts[1:])
    return lines


def serialize_save(entries):
    seen_sections = []
    for e in sorted(entries, key=lambda x: x.order):
        if e.section is not None and e.section not in seen_sections:
            seen_sections.append(e.section)

    out_lines = []

    globals_ = [e for e in entries if e.section is None]
    for e in sorted(globals_, key=lambda x: x.order):
        out_lines.extend(entry_to_lines(e))

    for sec in seen_sections:
        out_lines.append(f"[{sec}]")
        sec_entries = [e for e in entries if e.section == sec]
        for e in sorted(sec_entries, key=lambda x: x.order):
            out_lines.extend(entry_to_lines(e))

    return "\n".join(out_lines) + "\n"


# ----------------------------
# GUI
# ----------------------------

class SaveEditorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Encrypted save.data Editor (Excel-ish)")
        self.geometry("1120x760")

        self.filepath_encrypted: Optional[str] = None
        self.entries = []
        self.section_tabs = {}
        self.section_frames = {}
        self.changed_tag = "changed"

        self._cell_editor: Optional[ttk.Entry] = None
        self._editing_tree: Optional[ttk.Treeview] = None
        self._editing_iid: Optional[str] = None
        self._editing_col: Optional[str] = None
        self._editing_var: Optional[tk.StringVar] = None
        self._debounce_job = None

        self._last_mtime: Optional[float] = None
        self._watch_interval_ms = 1000

        self._build_ui()

        self.bind_all("<F2>", self._f2_global, add="+")
        self.bind_all("<Control-Return>", self._ctrl_enter_global, add="+")
        self.bind_all("<Control-KP_Enter>", self._ctrl_enter_global, add="+")

        self.after(self._watch_interval_ms, self._watch_file_changes)

    # ---------- UI ----------

    def _build_ui(self):
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=8, pady=8)

        ttk.Button(bar, text="Open save.data (Encrypted)...", command=self.open_encrypted).pack(side="left")
        ttk.Button(bar, text="Save Encrypted (overwrite)", command=self.save_encrypted).pack(side="left", padx=(8, 0))
        ttk.Button(bar, text="Save Encrypted As...", command=self.save_encrypted_as).pack(side="left", padx=(8, 0))

        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=12)

        ttk.Button(bar, text="Add Row", command=self.add_row).pack(side="left")
        ttk.Button(bar, text="Delete Row(s)", command=self.delete_row).pack(side="left", padx=(8, 0))
        ttk.Button(bar, text="Delete ALL in Section", command=self.delete_all_in_section).pack(side="left", padx=(8, 0))

        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=12)

        ttk.Button(bar, text="Auto Waves...", command=self.auto_waves).pack(side="left")
        ttk.Button(bar, text="Copy Selected", command=self.copy_selected).pack(side="left", padx=(8, 0))
        ttk.Button(bar, text="Paste", command=self.paste_kv).pack(side="left", padx=(8, 0))

        self.status = ttk.Label(self, text="Open an encrypted save.data to begin.")
        self.status.pack(fill="x", padx=8, pady=(0, 6))

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=8, pady=8)

        hint = ttk.Label(
            self,
            text="Auto reload checks file changes every 1s. F2 edits. Ctrl+Enter fills selection. "
                 "Auto Waves ALWAYS appends a fresh w1..wN block (duplicates allowed)."
        )
        hint.pack(fill="x", padx=8, pady=(0, 8))

    # ---------- misc ----------

    def is_dirty(self) -> bool:
        return any(e.changed for e in self.entries)

    def update_status(self, extra: str = ""):
        changed = sum(1 for e in self.entries if e.changed)
        total = len(self.entries)
        path = self.filepath_encrypted if self.filepath_encrypted else "(no file)"
        msg = f"Encrypted file: {path} | Entries: {total} | Changed: {changed}"
        if extra:
            msg += " | " + extra
        self.status.config(text=msg)

    def clear_tabs(self):
        for tab_id in self.nb.tabs():
            self.nb.forget(tab_id)
        self.section_tabs.clear()
        self.section_frames.clear()

    def build_tabs(self):
        self.clear_tabs()

        sections = [None]
        for e in sorted(self.entries, key=lambda x: x.order):
            if e.section is not None and e.section not in sections:
                sections.append(e.section)

        for sec in sections:
            title = "GLOBAL" if sec is None else sec
            frame = ttk.Frame(self.nb)
            self.nb.add(frame, text=title)
            self.section_frames[sec] = frame

            tree = ttk.Treeview(frame, columns=("key", "value"), show="headings", selectmode="extended")
            tree.heading("key", text="Key")
            tree.heading("value", text="Value")
            tree.column("key", width=340, anchor="w")
            tree.column("value", width=720, anchor="w")
            tree.pack(fill="both", expand=True, side="left")

            vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
            vsb.pack(side="right", fill="y")
            tree.configure(yscrollcommand=vsb.set)

            tree.tag_configure(self.changed_tag)

            tree._anchor_iid = None
            tree._active_col = "#2"

            tree.bind("<Button-1>", self._on_single_click)
            tree.bind("<Double-1>", lambda ev, s=sec: self.on_double_click(ev, s))
            tree.bind("<MouseWheel>", self._dismiss_inline_editor)

            tree.bind("<Up>", self._nav_up)
            tree.bind("<Down>", self._nav_down)
            tree.bind("<Left>", self._nav_left)
            tree.bind("<Right>", self._nav_right)
            tree.bind("<Return>", self._enter_edit)
            tree.bind("<Delete>", lambda ev: self.delete_row())

            self.section_tabs[sec] = tree

        self.refresh_all_views()

    def refresh_all_views(self):
        self._dismiss_inline_editor()
        for sec, tree in self.section_tabs.items():
            tree.delete(*tree.get_children())
            sec_entries = [e for e in self.entries if e.section == sec]
            sec_entries.sort(key=lambda x: x.order)

            for e in sec_entries:
                shown_val = e.value.replace("\n", "\\n")
                tags = (self.changed_tag,) if e.changed else ()
                tree.insert("", "end", iid=str(e.order), values=(e.key, shown_val), tags=tags)

        self.update_status()

    def current_section_and_tree(self):
        title = self.nb.tab(self.nb.select(), "text")
        sec = None if title == "GLOBAL" else title
        return sec, self.section_tabs.get(sec)

    def get_entry_by_order(self, order: int):
        for e in self.entries:
            if e.order == order:
                return e
        return None

    # ---------- file open/reload ----------

    def open_encrypted(self):
        path = filedialog.askopenfilename(
            title="Open encrypted save.data",
            filetypes=[("Save data", "*.data *.bin *.*"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            self._load_encrypted_path(path, user_initiated=True)
        except Exception as ex:
            messagebox.showerror("Open failed", f"Could not decrypt:\n{ex}")

    def _load_encrypted_path(self, path: str, user_initiated: bool = False):
        data = decrypt_save_file(path)
        text = data.decode("latin-1")

        self.filepath_encrypted = path
        self.entries = parse_save_text(text)
        self.build_tabs()

        try:
            self._last_mtime = os.path.getmtime(path)
        except OSError:
            self._last_mtime = None

        if user_initiated:
            messagebox.showinfo("Decrypted", "Decrypted successfully. You can now edit and re-encrypt.")
        self.update_status("Loaded/decrypted")

    def _watch_file_changes(self):
        try:
            if self.filepath_encrypted and os.path.exists(self.filepath_encrypted):
                mtime = os.path.getmtime(self.filepath_encrypted)
                if self._last_mtime is None:
                    self._last_mtime = mtime
                elif mtime != self._last_mtime:
                    self._last_mtime = mtime
                    if self.is_dirty():
                        ans = messagebox.askyesno(
                            "File changed on disk",
                            "save.data changed externally, but you have unsaved edits.\n\n"
                            "Reload from disk and discard your current edits?"
                        )
                        if ans:
                            self._load_encrypted_path(self.filepath_encrypted, user_initiated=False)
                        else:
                            self.update_status("Disk changed (not reloaded)")
                    else:
                        self._load_encrypted_path(self.filepath_encrypted, user_initiated=False)
        except Exception:
            pass
        finally:
            self.after(self._watch_interval_ms, self._watch_file_changes)

    # ---------- selection ----------

    def _tree_items(self, tree: ttk.Treeview):
        return list(tree.get_children(""))

    def _set_anchor(self, tree: ttk.Treeview, iid: str):
        tree._anchor_iid = iid

    def _get_anchor(self, tree: ttk.Treeview) -> Optional[str]:
        return getattr(tree, "_anchor_iid", None)

    def _select_range(self, tree: ttk.Treeview, start_iid: str, end_iid: str):
        items = self._tree_items(tree)
        if start_iid not in items or end_iid not in items:
            return
        a = items.index(start_iid)
        b = items.index(end_iid)
        lo, hi = (a, b) if a <= b else (b, a)
        rng = items[lo:hi + 1]
        tree.selection_set(rng)
        tree.focus(end_iid)
        tree.see(end_iid)

    def _get_active_col(self, tree: ttk.Treeview) -> str:
        return getattr(tree, "_active_col", "#2")

    def _set_active_col(self, tree: ttk.Treeview, col: str):
        if col in ("#1", "#2"):
            tree._active_col = col

    def _on_single_click(self, event):
        sec, tree = self.current_section_and_tree()
        if tree is None:
            return

        row = tree.identify_row(event.y)
        col = tree.identify_column(event.x)

        if not row:
            self._dismiss_inline_editor()
            return

        if col in ("#1", "#2"):
            self._set_active_col(tree, col)

        self._dismiss_inline_editor()

        shift = (event.state & 0x0001) != 0
        ctrl = (event.state & 0x0004) != 0

        anchor = self._get_anchor(tree)

        if shift and anchor:
            self._select_range(tree, anchor, row)
            return "break"

        if ctrl:
            cur = set(tree.selection())
            if row in cur:
                cur.remove(row)
            else:
                cur.add(row)
            tree.selection_set(list(cur))
            tree.focus(row)
            tree.see(row)
            self._set_anchor(tree, row)
            return "break"

        tree.selection_set((row,))
        tree.focus(row)
        tree.see(row)
        self._set_anchor(tree, row)
        return "break"

    # ---------- inline editing ----------

    def _dismiss_inline_editor(self, event=None):
        if self._debounce_job is not None:
            try:
                self.after_cancel(self._debounce_job)
            except Exception:
                pass
            self._debounce_job = None

        if self._cell_editor is not None:
            try:
                self._cell_editor.destroy()
            except Exception:
                pass

        self._cell_editor = None
        self._editing_tree = None
        self._editing_iid = None
        self._editing_col = None
        self._editing_var = None

    def on_double_click(self, event, section):
        tree = self.section_tabs[section]
        if tree.identify("region", event.x, event.y) != "cell":
            return

        iid = tree.identify_row(event.y)
        col = tree.identify_column(event.x)
        if not iid or col not in ("#1", "#2"):
            return

        self._set_active_col(tree, col)

        entry = self.get_entry_by_order(int(iid))
        if not entry:
            return

        if col == "#2" and "\n" in entry.value:
            self._edit_multiline(entry)
            return

        self._start_inline_edit(tree, iid, col)

    def _start_inline_edit(self, tree: ttk.Treeview, iid: str, col: str):
        self._dismiss_inline_editor()

        bbox = tree.bbox(iid, col)
        if not bbox:
            return
        x, y, w, h = bbox

        values = tree.item(iid, "values")
        current = values[0] if col == "#1" else values[1].replace("\\n", "\n")

        var = tk.StringVar(value=current)
        editor = ttk.Entry(tree, textvariable=var)
        editor.place(x=x, y=y, width=w, height=h)
        editor.focus_set()
        editor.icursor("end")

        self._cell_editor = editor
        self._editing_tree = tree
        self._editing_iid = iid
        self._editing_col = col
        self._editing_var = var

        var.trace_add("write", lambda *_: self._schedule_live_commit())

        editor.bind("<Escape>", lambda e: (self._dismiss_inline_editor(), "break"))
        editor.bind("<Return>", lambda e: (self._commit_inline_edit(final=True), self._dismiss_inline_editor(), "break"))
        editor.bind("<FocusOut>", lambda e: (self._commit_inline_edit(final=True), self._dismiss_inline_editor()))
        editor.bind("<Control-Return>", lambda e: (self.fill_ctrl_enter(), "break"))
        editor.bind("<Control-KP_Enter>", lambda e: (self.fill_ctrl_enter(), "break"))

    def _schedule_live_commit(self):
        if self._debounce_job is not None:
            try:
                self.after_cancel(self._debounce_job)
            except Exception:
                pass
        self._debounce_job = self.after(60, lambda: self._commit_inline_edit(final=False))

    def _commit_inline_edit(self, final: bool):
        if not (self._cell_editor and self._editing_tree and self._editing_iid and self._editing_col and self._editing_var):
            return

        iid = self._editing_iid
        col = self._editing_col
        new_text = self._editing_var.get()

        entry = self.get_entry_by_order(int(iid))
        if not entry:
            return

        if col == "#1":
            self._apply_key(entry, new_text)
        else:
            entry.value = new_text.replace("\\n", "\n")

        vals = list(self._editing_tree.item(iid, "values"))
        if col == "#1":
            vals[0] = entry.key
        else:
            vals[1] = entry.value.replace("\n", "\\n")
        self._editing_tree.item(iid, values=tuple(vals))

        tags = (self.changed_tag,) if entry.changed else ()
        self._editing_tree.item(iid, tags=tags)

        self.update_status("Editing...")
        if final:
            self.update_status()

    def _apply_key(self, entry: Entry, new_key: str):
        # duplicates allowed (per your request)
        new_key = (new_key or "").strip()
        if not new_key:
            return
        entry.key = new_key

    def _edit_multiline(self, entry: Entry):
        win = tk.Toplevel(self)
        win.title("Edit Value (multiline)")
        win.geometry("700x420")
        win.transient(self)
        win.grab_set()

        txt = tk.Text(win, wrap="word", height=14)
        txt.pack(fill="both", expand=True, padx=10, pady=10)
        txt.insert("1.0", entry.value)

        def apply():
            entry.value = txt.get("1.0", "end-1c")
            self.refresh_all_views()
            win.destroy()

        btns = ttk.Frame(win)
        btns.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side="right")
        ttk.Button(btns, text="Apply", command=apply).pack(side="right", padx=(0, 8))

    # ---------- keyboard ----------

    def _nav_up(self, event):
        self._dismiss_inline_editor()
        self._move_focus(-1)
        return "break"

    def _nav_down(self, event):
        self._dismiss_inline_editor()
        self._move_focus(+1)
        return "break"

    def _nav_left(self, event):
        sec, tree = self.current_section_and_tree()
        if tree:
            self._set_active_col(tree, "#1")
        return "break"

    def _nav_right(self, event):
        sec, tree = self.current_section_and_tree()
        if tree:
            self._set_active_col(tree, "#2")
        return "break"

    def _enter_edit(self, event):
        return self._edit_focused_cell()

    def _f2_global(self, event):
        return self._edit_focused_cell()

    def _ctrl_enter_global(self, event):
        self.fill_ctrl_enter()
        return "break"

    def _edit_focused_cell(self):
        sec, tree = self.current_section_and_tree()
        if tree is None:
            return "break"

        items = tree.get_children("")
        if not items:
            return "break"

        if self._cell_editor is not None:
            self._commit_inline_edit(final=True)
            self._dismiss_inline_editor()

        focus = tree.focus()
        if not focus:
            focus = tree.selection()[0] if tree.selection() else items[0]
            tree.focus(focus)
            tree.see(focus)

        col = self._get_active_col(tree)
        entry = self.get_entry_by_order(int(focus))
        if entry and col == "#2" and "\n" in entry.value:
            self._edit_multiline(entry)
            return "break"

        self._start_inline_edit(tree, focus, col)
        return "break"

    def _move_focus(self, direction: int):
        sec, tree = self.current_section_and_tree()
        if tree is None:
            return
        items = list(tree.get_children(""))
        if not items:
            return

        focus = tree.focus() or items[0]
        try:
            idx = items.index(focus)
        except ValueError:
            idx = 0

        idx2 = max(0, min(len(items) - 1, idx + direction))
        nxt = items[idx2]
        tree.focus(nxt)
        tree.see(nxt)
        self._set_anchor(tree, nxt)

    # ---------- Ctrl+Enter fill ----------

    def fill_ctrl_enter(self):
        sec, tree = self.current_section_and_tree()
        if tree is None:
            return

        sel = list(tree.selection())
        if not sel:
            return

        self._commit_inline_edit(final=False)

        focus = tree.focus()
        if not focus:
            return

        col = self._get_active_col(tree)
        src_entry = self.get_entry_by_order(int(focus))
        if not src_entry:
            return

        fill_text = src_entry.key if col == "#1" else src_entry.value

        for iid in sel:
            e = self.get_entry_by_order(int(iid))
            if not e:
                continue
            if col == "#1":
                e.key = fill_text
            else:
                e.value = fill_text

        self.refresh_all_views()
        self.update_status("Filled selection (Ctrl+Enter)")

    # ---------- copy/paste ----------

    def copy_selected(self):
        sec, tree = self.current_section_and_tree()
        if tree is None:
            return
        sel = tree.selection()
        if not sel:
            return
        sel_orders = sorted(int(x) for x in sel)
        lines = []
        for o in sel_orders:
            e = self.get_entry_by_order(o)
            if not e:
                continue
            v = e.value.replace("\n", "\\n")
            lines.append(f"{e.key}\t{v}")
        self.clipboard_clear()
        self.clipboard_append("\n".join(lines))

    def paste_kv(self):
        try:
            clip = self.clipboard_get()
        except tk.TclError:
            return

        sec, tree = self.current_section_and_tree()
        if tree is None:
            return

        rows = [r for r in clip.splitlines() if r.strip()]
        if not rows:
            return

        parsed = []
        two_col = False
        for r in rows:
            if "\t" in r:
                k, v = r.split("\t", 1)
                parsed.append((k.strip(), v))
                two_col = True
            else:
                parsed.append((None, r))

        sel = list(tree.selection())
        col = self._get_active_col(tree)

        if not sel:
            for k, v in parsed:
                self.add_row()
                new_e = max(self.entries, key=lambda x: x.order)
                if two_col and k:
                    new_e.key = k
                new_e.value = v
            self.refresh_all_views()
            return

        sel_orders = sorted(int(x) for x in sel)
        sel_entries = [self.get_entry_by_order(o) for o in sel_orders]
        sel_entries = [e for e in sel_entries if e is not None]

        if two_col:
            n = min(len(sel_entries), len(parsed))
            for i in range(n):
                k, v = parsed[i]
                if k:
                    sel_entries[i].key = k
                sel_entries[i].value = v
        else:
            n = min(len(sel_entries), len(parsed))
            for i in range(n):
                _, v = parsed[i]
                if col == "#1":
                    sel_entries[i].key = v
                else:
                    sel_entries[i].value = v

        self.refresh_all_views()
        self.update_status("Pasted")

    # ---------- add/delete ----------

    def add_row(self):
        sec, tree = self.current_section_and_tree()
        if tree is None:
            return

        next_order = (max((e.order for e in self.entries), default=-1) + 1)
        indent = "" if sec is None else "\t"

        e = Entry(sec, indent, "newkey", "", next_order)
        self.entries.append(e)
        self.refresh_all_views()

        tree.selection_set(str(e.order))
        tree.focus(str(e.order))
        tree.see(str(e.order))
        self._set_anchor(tree, str(e.order))
        self._set_active_col(tree, "#2")
        self._start_inline_edit(tree, str(e.order), "#2")

    def delete_row(self):
        sec, tree = self.current_section_and_tree()
        if tree is None:
            return
        sel = tree.selection()
        if not sel:
            return
        if not messagebox.askyesno("Delete", f"Delete {len(sel)} selected row(s)?"):
            return
        orders = set(int(x) for x in sel)
        self.entries = [e for e in self.entries if e.order not in orders]
        self.refresh_all_views()

    def delete_all_in_section(self):
        sec, tree = self.current_section_and_tree()
        if tree is None:
            return
        label = "GLOBAL" if sec is None else f"[{sec}]"
        if not messagebox.askyesno("Delete All", f"Delete ALL entries in {label}?"):
            return
        self.entries = [e for e in self.entries if e.section != sec]
        self.refresh_all_views()

    # ---------- Auto Waves (DUPLICATE BLOCKS ALLOWED) ----------

    def auto_waves(self):
        if not self.entries:
            messagebox.showerror("No data", "Open an encrypted save.data first.")
            return

        sec, _tree = self.current_section_and_tree()
        tab_name = self.nb.tab(self.nb.select(), "text").lower()
        sec_name = (sec or "").lower()
        if ("wave" not in tab_name) and ("wave" not in sec_name):
            messagebox.showerror("Not in Waves", "Switch to the Waves section/tab first (a tab containing 'wave').")
            return

        n = simpledialog.askinteger("Auto Waves", "Append a NEW block: w1..wN\nEnter N:", parent=self, minvalue=1)
        if not n:
            return

        wave_value = simpledialog.askstring("Auto Waves", "Value for ALL new w# keys (single line):", parent=self)
        if wave_value is None:
            return
        if "\n" in wave_value or "\r" in wave_value:
            messagebox.showerror("Single line only", "Auto Waves only accepts a single-line value.")
            return

        indent = "" if sec is None else "\t"
        max_order = max((e.order for e in self.entries), default=-1)

        # IMPORTANT: always append w1..wN, even if they already exist
        for i in range(1, n + 1):
            max_order += 1
            self.entries.append(Entry(sec, indent, f"w{i}", wave_value, max_order))

        # No sorting/reorder: keeps block order intact.
        self.refresh_all_views()
        self.update_status(f"Auto Waves: appended new block w1..w{n} (added {n})")

    # ---------- save encrypted (NO BACKUP) ----------

    def _build_plaintext_bytes(self) -> bytes:
        out_text = serialize_save(self.entries)
        return out_text.encode("latin-1")

    def save_encrypted(self):
        if not self.filepath_encrypted:
            messagebox.showerror("No file", "Open an encrypted save.data first.")
            return
        try:
            data = self._build_plaintext_bytes()
            encrypt_save_file(data, self.filepath_encrypted)

            try:
                self._last_mtime = os.path.getmtime(self.filepath_encrypted)
            except OSError:
                self._last_mtime = None

            for e in self.entries:
                e._orig_key = e.key
                e._orig_value = e.value

            self.refresh_all_views()
            messagebox.showinfo("Saved", "Encrypted and saved (no backup created).")
        except Exception as ex:
            messagebox.showerror("Save failed", str(ex))

    def save_encrypted_as(self):
        if not self.entries:
            messagebox.showerror("No data", "Open an encrypted save.data first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save encrypted save.data As...",
            defaultextension=".data",
            filetypes=[("Save data", "*.data *.bin *.*"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            data = self._build_plaintext_bytes()
            encrypt_save_file(data, path)
            messagebox.showinfo("Saved As", f"Encrypted save written to:\n{path}")
        except Exception as ex:
            messagebox.showerror("Save As failed", str(ex))


if __name__ == "__main__":
    app = SaveEditorGUI()
    app.mainloop()
