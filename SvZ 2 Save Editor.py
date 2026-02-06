#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SvZ Defense 2 - Local Save Editor (binary SAVE + gzip + obfuscated CRC)
"""

from __future__ import annotations

import gzip
import io
import re
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ---- Optional drag & drop ----
DND_AVAILABLE = False
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES  # type: ignore
    DND_AVAILABLE = True
except Exception:
    DND_AVAILABLE = False

# ---- File format bits ----
SAVE_MAGIC = b"SAVE"
HEADER_LEN = 14
GZIP_MAGIC = b"\x1f\x8b"
CRC_SALT = bytes([250, 87, 240, 13])

# ---- Text parsing ----
KV_RE = re.compile(r"^(\s*)([^=\[\]\r\n]+?)\s*=\s*(.*?)\s*$")
SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")


# ================= Unity RNG parity (Xorshift128 + InitSeed) =================
def _u32(x: int) -> int:
    return x & 0xFFFFFFFF


def unity_xorshift_init_seed(seed_i32: int) -> list[int]:
    MT = 1812433253
    x = _u32(seed_i32)
    y = _u32(MT * x + 1)
    z = _u32(MT * y + 1)
    w = _u32(MT * z + 1)
    return [x, y, z, w]


def unity_xorshift_next_u32(state: list[int]) -> int:
    x, y, z, w = state
    t = _u32(x ^ _u32(x << 11))
    x, y, z = y, z, w
    w = _u32(_u32(w ^ (w >> 19)) ^ _u32(t ^ (t >> 8)))
    state[:] = [x, y, z, w]
    return w


def unity_random_range_int(seed_i32: int, min_incl: int, max_excl: int) -> int:
    span = max_excl - min_incl
    if span <= 0:
        return min_incl
    st = unity_xorshift_init_seed(seed_i32)
    r = unity_xorshift_next_u32(st)
    return int(min_incl + (r % span))


# ================= Header + CRC helpers =================
def parse_header(h: bytes) -> dict:
    if len(h) != HEADER_LEN:
        raise ValueError("Bad header length")
    if h[:4] != SAVE_MAGIC:
        raise ValueError(f"Missing SAVE magic. First 4 bytes: {h[:4]!r}")
    return {"crc_be": int.from_bytes(h[6:10], "big")}


def set_header_crc(header: bytearray, crc_u32: int) -> None:
    header[6:10] = int(crc_u32 & 0xFFFFFFFF).to_bytes(4, "big")


def revert_crc_for_calc(file_bytes: bytes) -> bytes:
    b = bytearray(file_bytes)
    b[6:10] = CRC_SALT
    return bytes(b)


def compute_obfuscated_crc(file_bytes: bytes) -> int:
    salted = revert_crc_for_calc(file_bytes)
    crc = zlib.crc32(salted) & 0xFFFFFFFF
    seed_i32 = struct.unpack("<i", struct.pack("<I", crc))[0]
    num2 = unity_random_range_int(seed_i32, 0, 2**31 - 1)
    return (crc ^ (num2 & 0xFFFFFFFF)) & 0xFFFFFFFF


# ================= gzip helpers =================
def find_gzip_offset(raw: bytes) -> int:
    idx = raw.find(GZIP_MAGIC)
    if idx == -1:
        raise ValueError("No gzip stream found in file (missing 1F 8B).")
    return idx


def gzip_decompress(blob: bytes) -> bytes:
    with gzip.GzipFile(fileobj=io.BytesIO(blob), mode="rb") as gz:
        return gz.read()


def gzip_compress(data: bytes) -> bytes:
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb") as gz:
        gz.write(data)
    return out.getvalue()


# ================= Save text region extraction =================
def extract_text_region(payload: bytes) -> tuple[int, bytes]:
    best = None
    best_score = -1.0
    for m in re.finditer(b"timestamp", payload):
        start = m.start()
        tail = payload[start:]
        printable = sum((32 <= b <= 126) or b in (9, 10, 13) for b in tail)
        score = printable / max(1, len(tail))
        if score > best_score:
            best = start
            best_score = score

    if best is None:
        m = re.search(rb"[ -~\t\r\n]{200,}", payload)
        if not m:
            raise ValueError("Could not locate a text region in decompressed payload.")
        best = m.start()

    return best, payload[best:]


@dataclass
class LineKV:
    section: str
    key: str
    val: str
    line_index: int
    indent: str


def parse_save_text(text: str) -> tuple[list[str], dict[str, list[LineKV]]]:
    lines = text.splitlines(True)
    by_sec: dict[str, list[LineKV]] = {"": []}
    cur = ""
    for i, raw_line in enumerate(lines):
        line = raw_line.rstrip("\r\n")
        sm = SECTION_RE.match(line)
        if sm:
            cur = sm.group(1).strip()
            by_sec.setdefault(cur, [])
            continue
        km = KV_RE.match(line)
        if km:
            indent, key, val = km.groups()
            key = key.strip()
            if key:
                by_sec.setdefault(cur, [])
                by_sec[cur].append(LineKV(cur, key, val, i, indent))
    return lines, by_sec


def apply_changes(lines: list[str],
                  edits: dict[tuple[str, str], tuple[str, str]],
                  deletes: set[tuple[str, str]]) -> list[str]:
    out = list(lines)

    section_for_line = [""] * len(out)
    cur = ""
    for i in range(len(out)):
        s = out[i].rstrip("\r\n")
        sm = SECTION_RE.match(s)
        if sm:
            cur = sm.group(1).strip()
        section_for_line[i] = cur

    for idx in range(len(out)):
        line = out[idx].rstrip("\r\n")
        km = KV_RE.match(line)
        if not km:
            continue
        indent, key, _val = km.groups()
        key = key.strip()
        sec = section_for_line[idx]

        if (sec, key) in deletes:
            out[idx] = ""
            continue

        if (sec, key) in edits:
            new_key, new_val = edits[(sec, key)]
            new_key = new_key.strip() or key
            line_ending = "\r\n" if out[idx].endswith("\r\n") else ("\n" if out[idx].endswith("\n") else "")
            out[idx] = f"{indent}{new_key} = {new_val}{line_ending}"

    out = [ln for ln in out if ln != ""]
    return out


def insert_new_entries(lines: list[str], entries: list[tuple[str, str, str]]) -> list[str]:
    if not entries:
        return list(lines)

    out = list(lines)

    nl = "\n"
    for ln in out:
        if ln.endswith("\r\n"):
            nl = "\r\n"
            break

    cur = ""
    last_idx: dict[str, int] = {"": -1}
    first_section_idx = None

    for i, raw in enumerate(out):
        s = raw.rstrip("\r\n")
        sm = SECTION_RE.match(s)
        if sm:
            cur = sm.group(1).strip()
            if first_section_idx is None:
                first_section_idx = i
            last_idx.setdefault(cur, i)
            continue
        last_idx[cur] = i

    by_sec: dict[str, list[tuple[str, str]]] = {}
    for sec, k, v in entries:
        sec = sec or ""
        by_sec.setdefault(sec, []).append((k, v))

    existing_secs = []
    seen = set()
    cur = ""
    for raw in out:
        s = raw.rstrip("\r\n")
        sm = SECTION_RE.match(s)
        if sm:
            cur = sm.group(1).strip()
            if cur not in seen:
                existing_secs.append(cur)
                seen.add(cur)

    sec_order = []
    if "" in by_sec:
        sec_order.append("")
    for s in existing_secs:
        if s in by_sec and s not in sec_order:
            sec_order.append(s)
    for s in sorted(by_sec.keys()):
        if s not in sec_order:
            sec_order.append(s)

    for sec in sec_order:
        pairs = by_sec.get(sec, [])
        if not pairs:
            continue

        if sec == "":
            ins = first_section_idx if first_section_idx is not None else len(out)
            indent = ""
        else:
            indent = "\t"
            if sec not in last_idx:
                if out and not out[-1].endswith("\n") and not out[-1].endswith("\r\n"):
                    out[-1] += nl
                out.append(f"[{sec}]{nl}")
                last_idx[sec] = len(out) - 1
            ins = last_idx[sec] + 1

        new_lines = [f"{indent}{k} = {v}{nl}" for k, v in pairs]
        out[ins:ins] = new_lines

        for s in list(last_idx.keys()):
            if last_idx[s] >= ins:
                last_idx[s] += len(new_lines)
        if first_section_idx is not None and ins <= first_section_idx:
            first_section_idx += len(new_lines)

    return out


# ================= GUI Model =================
@dataclass
class RowModel:
    section: str
    orig_key: str
    orig_val: str
    key: str
    val: str
    is_new: bool = False
    deleted: bool = False


class SaveEditorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("SvZ Defense 2 - Local Save Editor")
        root.geometry("1280x820")

        self.in_path: Path | None = None
        self.last_mtime: float | None = None

        self.raw_file: bytes | None = None
        self.header: bytearray | None = None
        self.payload: bytes | None = None
        self.text_off: int | None = None
        self.lines: list[str] | None = None
        self.by_section: dict[str, list[LineKV]] | None = None

        self.tree: ttk.Treeview
        self.model: dict[str, RowModel] = {}
        self.section_nodes: dict[str, str] = {}
        self.section_order: list[str] = []
        self._filter_detached: dict[str, tuple[str, int]] = {}
        self._search_prev_open: dict[str, bool] = {}
        self._search_active_prev = False

        self._editor: ttk.Entry | None = None
        self._editor_iid: str | None = None
        self._editor_col: str | None = None  # "key" or "val"

        self.search_var = tk.StringVar(value="")
        self.filter_keys = tk.BooleanVar(value=True)
        self.filter_vals = tk.BooleanVar(value=True)
        self.auto_reload = tk.BooleanVar(value=True)
        self.status = tk.StringVar(
            value=("Drag & drop 'local' here." if DND_AVAILABLE else "Tip: pip install tkinterdnd2 for drag & drop")
        )

        self._build_ui()
        self.root.after(800, self._watch_file)

    # ---------------- UI ----------------
    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        ttk.Button(top, text="Open local…", command=self.open_file).pack(side="left")
        ttk.Button(top, text="Save patched as…", command=self.save_patched).pack(side="left", padx=(8, 0))

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Button(top, text="Add key/value", command=self.add_empty_row).pack(side="left")
        ttk.Button(top, text="Delete ALL in section", command=self.delete_all_in_section).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Auto Waves…", command=self.auto_waves_dialog).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Unmark deletes", command=self.unmark_deletes).pack(side="left", padx=(8, 0))

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Label(top, text="Search:").pack(side="left")
        ent = ttk.Entry(top, textvariable=self.search_var, width=34)
        ent.pack(side="left", padx=(6, 8))
        self.search_var.trace_add("write", lambda *_: self.apply_filter())

        ttk.Checkbutton(top, text="keys", variable=self.filter_keys, command=self.apply_filter).pack(side="left")
        ttk.Checkbutton(top, text="values", variable=self.filter_vals, command=self.apply_filter).pack(side="left")

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Checkbutton(top, text="Auto-reload", variable=self.auto_reload).pack(side="left")

        ttk.Label(top, textvariable=self.status).pack(side="left", padx=(16, 0))

        mid = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        mid.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(
            mid,
            columns=("key", "val", "del"),
            show="tree headings",
            selectmode="extended",
        )
        vsb = ttk.Scrollbar(mid, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(mid, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.heading("#0", text="Section")
        self.tree.heading("key", text="Key")
        self.tree.heading("val", text="Value")
        self.tree.heading("del", text="Delete")

        self.tree.column("#0", width=240, stretch=False)
        self.tree.column("key", width=320, stretch=False)
        self.tree.column("val", width=720, stretch=True)
        self.tree.column("del", width=70, stretch=False, anchor="center")

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        mid.rowconfigure(0, weight=1)
        mid.columnconfigure(0, weight=1)

        # ---- Selection color: stronger contrast ----
        style = ttk.Style()
        # These colors are intentionally high-contrast for readability.
        style.map(
            "Treeview",
            background=[("selected", "#2b5fb8")],
            foreground=[("selected", "#ffffff")],
        )

        # ---- Tag styles (changed rows) ----
        # Use a warm background so "changed" stands out even when not selected.
        self.tree.tag_configure("changed", background="#fff3cd", foreground="#000000")
        self.tree.tag_configure("deleted", foreground="#888888")
        self.tree.tag_configure("newrow", background="#e7f7ee", foreground="#000000")

        # Bindings
        self.tree.bind("<Double-1>", self._start_edit_from_event)
        self.tree.bind("<F2>", self._start_edit_from_focus)
        self.tree.bind("<Return>", self._start_edit_from_focus)
        self.tree.bind("<Delete>", self._toggle_delete_selected)
        self.tree.bind("<Button-1>", self._handle_click_delete_column, add="+")
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._destroy_editor(commit=True))
        self.tree.bind("<Control-v>", self._paste_grid)

        # Drag & drop
        if DND_AVAILABLE:
            try:
                self.root.drop_target_register(DND_FILES)
                self.root.dnd_bind("<<Drop>>", self._on_drop)
            except Exception:
                pass

    # ---------------- Drag/drop ----------------
    def _parse_dnd_paths(self, data: str) -> list[str]:
        data = data.strip()
        out, buf = [], ""
        in_brace = False
        for ch in data:
            if ch == "{":
                in_brace = True
                buf = ""
            elif ch == "}":
                in_brace = False
                out.append(buf)
                buf = ""
            elif ch.isspace() and not in_brace:
                if buf:
                    out.append(buf)
                    buf = ""
            else:
                buf += ch
        if buf:
            out.append(buf)
        return out

    def _on_drop(self, event):
        paths = self._parse_dnd_paths(event.data)
        if not paths:
            return
        p = Path(paths[0])
        if p.exists():
            try:
                self._load(p, reason="manual")
            except Exception as e:
                messagebox.showerror("Open failed", str(e))

    # ---------------- Open/load ----------------
    def open_file(self):
        p = filedialog.askopenfilename(title="Select 'local' save file", filetypes=[("All files", "*.*")])
        if not p:
            return
        try:
            self._load(Path(p), reason="manual")
        except Exception as e:
            messagebox.showerror("Open failed", str(e))

    def _capture_open_sections(self) -> dict[str, bool]:
        out = {}
        for sec, iid in self.section_nodes.items():
            try:
                out[sec] = bool(self.tree.item(iid, "open"))
            except Exception:
                out[sec] = False
        return out

    def _restore_open_sections(self, state: dict[str, bool]):
        for sec, open_ in state.items():
            iid = self.section_nodes.get(sec)
            if iid:
                try:
                    self.tree.item(iid, open=open_)
                except Exception:
                    pass

    def _load(self, path: Path, reason: str):
        open_state = {}
        yview_top = None
        if reason == "reload":
            open_state = self._capture_open_sections()
            try:
                yview_top = self.tree.yview()[0]
            except Exception:
                yview_top = None

        raw = path.read_bytes()
        header = bytearray(raw[:HEADER_LEN])
        hinfo = parse_header(bytes(header))

        gz_off = find_gzip_offset(raw)
        payload = gzip_decompress(raw[gz_off:])

        text_off, text_bytes = extract_text_region(payload)
        text = text_bytes.decode("utf-8", errors="replace")
        lines, by_section = parse_save_text(text)

        computed = compute_obfuscated_crc(raw)
        ok = computed == hinfo["crc_be"]

        self.in_path = path
        self.last_mtime = path.stat().st_mtime
        self.raw_file = raw
        self.header = header
        self.payload = payload
        self.text_off = text_off
        self.lines = lines
        self.by_section = by_section

        self._rebuild_tree()

        if reason == "reload" and open_state:
            self._restore_open_sections(open_state)

        self.apply_filter()

        if reason == "reload" and yview_top is not None:
            self.root.after(1, lambda: self.tree.yview_moveto(yview_top))

        self.status.set(f"Loaded: {path.name} | text@{text_off} | CRC {'OK' if ok else 'MISMATCH'}")

    def _rebuild_tree(self):
        self._destroy_editor(commit=True)
        self.tree.delete(*self.tree.get_children())
        self.model.clear()
        self.section_nodes.clear()
        self.section_order.clear()
        self._filter_detached.clear()
        self._search_prev_open.clear()
        self._search_active_prev = False

        if not self.by_section:
            return

        # section order by appearance
        seen = set()
        order = [""]
        cur = ""
        if self.lines:
            for raw in self.lines:
                s = raw.rstrip("\r\n")
                sm = SECTION_RE.match(s)
                if sm:
                    cur = sm.group(1).strip()
                    if cur not in seen:
                        order.append(cur)
                        seen.add(cur)
        for sec in self.by_section.keys():
            if sec != "" and sec not in seen:
                order.append(sec)
                seen.add(sec)

        for sec in order:
            title = "(root)" if sec == "" else f"[{sec}]"
            iid = f"SEC::{sec}"
            self.section_nodes[sec] = iid
            self.section_order.append(sec)
            self.tree.insert("", "end", iid=iid, text=title, values=("", "", ""))
            self.tree.item(iid, open=False)

            for kv in self.by_section.get(sec, []):
                row_iid = f"ROW::{sec}::{kv.key}::{kv.line_index}"
                self.model[row_iid] = RowModel(
                    section=sec,
                    orig_key=kv.key,
                    orig_val=kv.val,   # IMPORTANT: store original value NOW
                    key=kv.key,
                    val=kv.val,
                    is_new=False,
                    deleted=False,
                )
                self.tree.insert(iid, "end", iid=row_iid, text="", values=(kv.key, kv.val, ""))
                self._refresh_item_style(row_iid)

    # ---------------- In-place editing ----------------
    def _destroy_editor(self, commit: bool):
        if not self._editor:
            return
        try:
            if commit:
                self._commit_editor(fill_all=False)
        except Exception:
            pass
        try:
            self._editor.destroy()
        except Exception:
            pass
        self._editor = None
        self._editor_iid = None
        self._editor_col = None

    def _start_edit_from_event(self, event):
        iid = self.tree.identify_row(event.y)
        col = self.tree.identify_column(event.x)

        if not iid or iid.startswith("SEC::") or iid not in self.model:
            return
        if self.model[iid].deleted:
            return

        if col == "#1":
            self._start_cell_edit(iid, "key")
        elif col == "#2":
            self._start_cell_edit(iid, "val")

    def _start_edit_from_focus(self, _event=None):
        sel = self.tree.selection()
        if not sel:
            return
        iid = sel[0]
        if iid.startswith("SEC::") or iid not in self.model:
            return
        self._start_cell_edit(iid, "val")

    def _start_cell_edit(self, iid: str, which: str):
        self._destroy_editor(commit=True)

        col_id = "#1" if which == "key" else "#2"
        bbox = self.tree.bbox(iid, col_id)
        if not bbox:
            return

        x, y, w, h = bbox
        current_text = self.tree.set(iid, which)

        ent = ttk.Entry(self.tree)
        ent.insert(0, current_text)
        ent.select_range(0, tk.END)
        ent.place(x=x, y=y, width=w, height=h)
        ent.focus_set()

        self._editor = ent
        self._editor_iid = iid
        self._editor_col = which

        ent.bind("<Return>", lambda _e: self._destroy_editor(commit=True))
        ent.bind("<Control-Return>", lambda _e: (self._commit_editor(fill_all=True), self._destroy_editor(commit=False)))
        ent.bind("<Escape>", lambda _e: self._destroy_editor(commit=False))
        ent.bind("<FocusOut>", lambda _e: self._destroy_editor(commit=True))

    def _commit_editor(self, fill_all: bool):
        if not self._editor or not self._editor_iid or not self._editor_col:
            return

        iid = self._editor_iid
        col = self._editor_col  # "key" or "val"
        new_text = self._editor.get()

        targets = [iid]
        if fill_all:
            targets = [x for x in self.tree.selection() if x in self.model and not self.model[x].deleted]

        for tid in targets:
            m = self.model.get(tid)
            if not m or m.deleted:
                continue

            if col == "key":
                nk = new_text.strip()
                if not nk:
                    continue
                m.key = nk
                self.tree.set(tid, "key", m.key)
            else:
                m.val = new_text
                self.tree.set(tid, "val", m.val)

            self._refresh_item_style(tid)

        self.apply_filter()

    # ---------------- Delete toggles ----------------
    def _handle_click_delete_column(self, event):
        iid = self.tree.identify_row(event.y)
        col = self.tree.identify_column(event.x)
        if not iid or iid.startswith("SEC::") or iid not in self.model:
            return
        if col != "#3":
            return
        self._toggle_delete_iid(iid)

    def _toggle_delete_selected(self, _event=None):
        sel = [x for x in self.tree.selection() if x in self.model]
        if not sel:
            return
        any_not_deleted = any(not self.model[i].deleted for i in sel)
        for iid in sel:
            self.model[iid].deleted = any_not_deleted
            self.tree.set(iid, "del", "X" if self.model[iid].deleted else "")
            self._refresh_item_style(iid)

    def _toggle_delete_iid(self, iid: str):
        m = self.model.get(iid)
        if not m:
            return
        m.deleted = not m.deleted
        self.tree.set(iid, "del", "X" if m.deleted else "")
        self._refresh_item_style(iid)

    def unmark_deletes(self):
        for iid, m in self.model.items():
            if m.deleted:
                m.deleted = False
                self.tree.set(iid, "del", "")
                self._refresh_item_style(iid)

    # ---------------- Highlight / styles ----------------
    def _refresh_item_style(self, iid: str):
        m = self.model.get(iid)
        if not m:
            return

        tags = set(self.tree.item(iid, "tags") or [])
        tags.discard("changed")
        tags.discard("deleted")
        tags.discard("newrow")

        if m.deleted:
            tags.add("deleted")
        else:
            if m.is_new:
                if (m.key.strip() != "") or (m.val != ""):
                    tags.add("changed")
                    tags.add("newrow")
            else:
                if (m.key != m.orig_key) or (m.val != m.orig_val):
                    tags.add("changed")

        self.tree.item(iid, tags=tuple(tags))

    # ---------------- Add rows ----------------
    def _current_section(self) -> str:
        sel = self.tree.selection()
        if not sel:
            return ""
        iid = sel[0]
        if iid.startswith("SEC::"):
            return iid[len("SEC::"):]
        parent = self.tree.parent(iid)
        if parent.startswith("SEC::"):
            return parent[len("SEC::"):]
        return ""

    def add_empty_row(self):
        if not self.lines:
            messagebox.showinfo("No file", "Open a save file first.")
            return

        sec = self._current_section()
        sec_node = self.section_nodes.get(sec)
        if not sec_node:
            sec_node = f"SEC::{sec}"
            title = "(root)" if sec == "" else f"[{sec}]"
            self.section_nodes[sec] = sec_node
            self.section_order.append(sec)
            self.tree.insert("", "end", iid=sec_node, text=title, values=("", "", ""))
            self.tree.item(sec_node, open=True)

        self.tree.item(sec_node, open=True)

        rid = f"NEW::{sec}::{len(self.model)}"
        self.model[rid] = RowModel(section=sec, orig_key="", orig_val="", key="", val="", is_new=True, deleted=False)
        self.tree.insert(sec_node, "end", iid=rid, text="", values=("", "", ""))
        self._refresh_item_style(rid)

        self.tree.selection_set(rid)
        self.tree.see(rid)
        self.root.after(1, lambda: self._start_cell_edit(rid, "key"))

    # ---------------- Delete all in section ----------------
    def delete_all_in_section(self):
        if not self.lines:
            messagebox.showinfo("No file", "Open a save file first.")
            return
        sec = self._current_section()
        title = "(root)" if sec == "" else f"[{sec}]"
        if not messagebox.askyesno("Delete section contents", f"Mark ALL rows in {title} for deletion?"):
            return
        sec_node = self.section_nodes.get(sec)
        if not sec_node:
            return
        for child in self.tree.get_children(sec_node):
            if child in self.model:
                self.model[child].deleted = True
                self.tree.set(child, "del", "X")
                self._refresh_item_style(child)

    # ---------------- Auto Waves ----------------
    def auto_waves_dialog(self):
        if not self.lines:
            messagebox.showinfo("No file", "Open a save file first.")
            return

        win = tk.Toplevel(self.root)
        win.title("Auto Waves")
        win.transient(self.root)
        win.grab_set()

        frm = ttk.Frame(win, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Add missing pairs into [waves]:").grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(frm, text="Max wave N (1..999):").grid(row=1, column=0, sticky="w", pady=(10, 2))
        max_var = tk.StringVar(value="999")
        ttk.Entry(frm, textvariable=max_var, width=10).grid(row=1, column=1, sticky="w", pady=(10, 2))

        ttk.Label(frm, text="Default wN value:").grid(row=2, column=0, sticky="w", pady=2)
        w_var = tk.StringVar(value="1")
        ttk.Entry(frm, textvariable=w_var, width=10).grid(row=2, column=1, sticky="w", pady=2)

        ttk.Label(frm, text="Default countN value:").grid(row=3, column=0, sticky="w", pady=2)
        c_var = tk.StringVar(value="1")
        ttk.Entry(frm, textvariable=c_var, width=10).grid(row=3, column=1, sticky="w", pady=2)

        def run():
            try:
                mx = int(max_var.get().strip())
                mx = max(1, min(999, mx))
                wv = w_var.get().strip()
                cv = c_var.get().strip()
            except Exception:
                messagebox.showerror("Invalid", "Enter valid numbers.")
                return
            added = self.auto_add_waves(mx, wv, cv)
            messagebox.showinfo("Auto Waves", f"Added {added} missing entries into [waves].")
            win.destroy()

        btns = ttk.Frame(frm)
        btns.grid(row=4, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side="right")
        ttk.Button(btns, text="Add", command=run).pack(side="right", padx=(0, 8))

    def auto_add_waves(self, max_wave: int, default_w: str, default_count: str) -> int:
        sec = "waves"
        sec_node = self.section_nodes.get(sec)
        if not sec_node:
            sec_node = f"SEC::{sec}"
            self.section_nodes[sec] = sec_node
            self.section_order.append(sec)
            self.tree.insert("", "end", iid=sec_node, text=f"[{sec}]", values=("", "", ""))
            self.tree.item(sec_node, open=True)

        existing_keys = set()
        for child in self.tree.get_children(sec_node):
            if child in self.model:
                k = self.model[child].key.strip()
                if k:
                    existing_keys.add(k)

        added = 0
        for n in range(1, max_wave + 1):
            wk = f"w{n}"
            ck = f"count{n}"

            if wk not in existing_keys:
                rid = f"NEW::{sec}::{len(self.model)}"
                self.model[rid] = RowModel(section=sec, orig_key="", orig_val="", key=wk, val=default_w, is_new=True, deleted=False)
                self.tree.insert(sec_node, "end", iid=rid, text="", values=(wk, default_w, ""))
                self._refresh_item_style(rid)
                existing_keys.add(wk)
                added += 1

            if ck not in existing_keys:
                rid = f"NEW::{sec}::{len(self.model)}"
                self.model[rid] = RowModel(section=sec, orig_key="", orig_val="", key=ck, val=default_count, is_new=True, deleted=False)
                self.tree.insert(sec_node, "end", iid=rid, text="", values=(ck, default_count, ""))
                self._refresh_item_style(rid)
                existing_keys.add(ck)
                added += 1

        self.tree.item(sec_node, open=True)
        self.apply_filter()
        return added

    # ---------------- Search / Filter ----------------
    def apply_filter(self):
        q = self.search_var.get().strip().lower()
        searching = bool(q)

        if searching and not self._search_active_prev:
            self._search_prev_open = {sec: bool(self.tree.item(iid, "open"))
                                     for sec, iid in self.section_nodes.items()}

        if (not searching) and self._search_active_prev:
            for sec, was_open in self._search_prev_open.items():
                iid = self.section_nodes.get(sec)
                if iid:
                    self.tree.item(iid, open=was_open)
            self._search_prev_open.clear()

        self._search_active_prev = searching

        if self._filter_detached:
            for iid, (parent, index) in list(self._filter_detached.items()):
                try:
                    self.tree.reattach(iid, parent, index)
                except Exception:
                    pass
            self._filter_detached.clear()

        if not searching:
            return

        in_keys = self.filter_keys.get()
        in_vals = self.filter_vals.get()

        keep = set()
        keep_sections = set()

        for iid, m in self.model.items():
            k = m.key.lower()
            v = m.val.lower()
            hay = []
            if in_keys:
                hay.append(k)
            if in_vals:
                hay.append(v)
            if any(q in h for h in hay):
                keep.add(iid)
                keep_sections.add(m.section)

        for iid, m in self.model.items():
            if iid in keep:
                continue
            parent = self.tree.parent(iid)
            try:
                idx = self.tree.index(iid)
                self.tree.detach(iid)
                self._filter_detached[iid] = (parent, idx)
            except Exception:
                pass

        for sec in keep_sections:
            sec_iid = self.section_nodes.get(sec)
            if sec_iid:
                self.tree.item(sec_iid, open=True)

    # ---------------- Paste grid ----------------
    def _paste_grid(self, _event=None):
        self._destroy_editor(commit=True)

        try:
            clip = self.root.clipboard_get()
        except Exception:
            return

        clip = clip.replace("\r\n", "\n").replace("\r", "\n")
        if not clip.strip():
            return

        rows = [r for r in clip.split("\n") if r != ""]
        grid = [r.split("\t") for r in rows]

        sel = [x for x in self.tree.selection() if x in self.model and not self.model[x].deleted]
        if not sel:
            return

        try:
            sel_sorted = sorted(sel, key=lambda iid: self.tree.index(iid))
        except Exception:
            sel_sorted = sel
        start_iid = sel_sorted[0]

        start_sec = self.model[start_iid].section
        sec_node = self.section_nodes.get(start_sec)
        if not sec_node:
            return

        visible_rows = [iid for iid in self.tree.get_children(sec_node) if iid in self.model and not self.model[iid].deleted]
        if start_iid not in visible_rows:
            return
        start_idx = visible_rows.index(start_iid)

        one_col = all(len(r) == 1 for r in grid)

        for r_i, cols in enumerate(grid):
            tgt_idx = start_idx + r_i
            if tgt_idx >= len(visible_rows):
                break
            iid = visible_rows[tgt_idx]
            m = self.model[iid]

            if one_col:
                m.val = cols[0]
                self.tree.set(iid, "val", m.val)
            else:
                if len(cols) >= 1 and cols[0].strip():
                    m.key = cols[0].strip()
                    self.tree.set(iid, "key", m.key)
                if len(cols) >= 2:
                    m.val = cols[1]
                    self.tree.set(iid, "val", m.val)

            self._refresh_item_style(iid)

        self.apply_filter()

    # ---------------- Save patched ----------------
    def _collect_changes(self):
        edits: dict[tuple[str, str], tuple[str, str]] = {}
        deletes: set[tuple[str, str]] = set()
        adds: list[tuple[str, str, str]] = []

        for _iid, m in self.model.items():
            if m.deleted:
                if (not m.is_new) and m.orig_key:
                    deletes.add((m.section, m.orig_key))
                continue

            k = m.key.strip()
            v = m.val

            if m.is_new:
                if k:
                    adds.append((m.section, k, v))
                continue

            if m.orig_key:
                if (k and k != m.orig_key) or (v != m.orig_val):
                    edits[(m.section, m.orig_key)] = (k or m.orig_key, v)

        return edits, deletes, adds

    def save_patched(self):
        if not self.in_path or self.raw_file is None:
            messagebox.showinfo("No file", "Open a 'local' save file first.")
            return

        out_str = filedialog.asksaveasfilename(
            title="Save patched file as",
            initialfile=self.in_path.name + ".patched",
            filetypes=[("All files", "*.*")],
        )
        if not out_str:
            return

        try:
            edits, deletes, adds = self._collect_changes()

            assert self.payload is not None
            assert self.text_off is not None
            assert self.lines is not None
            assert self.header is not None

            new_lines = apply_changes(self.lines, edits, deletes)
            new_lines = insert_new_entries(new_lines, adds)

            new_text = "".join(new_lines).encode("utf-8")
            new_payload = self.payload[: self.text_off] + new_text
            new_gz = gzip_compress(new_payload)

            out_bytes = bytes(self.header) + new_gz
            new_crc = compute_obfuscated_crc(out_bytes)

            header2 = bytearray(self.header)
            set_header_crc(header2, new_crc)
            out_bytes = bytes(header2) + new_gz

            Path(out_str).write_bytes(out_bytes)
            messagebox.showinfo("Saved", f"Saved patched file.\nCRC: {new_crc:08X}")

        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    # ---------------- Auto reload ----------------
    def _watch_file(self):
        try:
            if self.auto_reload.get() and self.in_path and self.in_path.exists():
                mtime = self.in_path.stat().st_mtime
                if self.last_mtime is not None and mtime != self.last_mtime:
                    self.last_mtime = mtime
                    self._load(self.in_path, reason="reload")
        except Exception:
            pass
        finally:
            self.root.after(800, self._watch_file)


def main():
    root = TkinterDnD.Tk() if DND_AVAILABLE else tk.Tk()
    SaveEditorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
