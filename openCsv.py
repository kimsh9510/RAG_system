import csv
import sys
import argparse
import itertools
from pathlib import Path

# raise CSV parser field size limit (helps with very large fields)
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(10 ** 9)

# optional encoding detector
try:
    import chardet # type: ignore
except Exception:
    chardet = None

try:
    import pandas as pd
    from tabulate import tabulate # type: ignore
except Exception:
    pd = None
    tabulate = None


def _truncate(val, maxw):
    s = "" if val is None else str(val)
    return s if len(s) <= maxw else s[: maxw - 3] + "..."


def _drop_blank_rows_df(df):
    # consider a row blank if all values are empty after stripping
    mask = ~(df.astype(str).apply(lambda col: col.str.strip()).eq("").all(axis=1))
    return df.loc[mask]


def detect_encoding_from_bytes(sample_bytes, prefer=None):
    # prefer is a list of encodings to try first
    if not sample_bytes:
        return None
    if chardet:
        try:
            det = chardet.detect(sample_bytes)
            enc = det.get("encoding")
            if enc:
                return enc
        except Exception:
            pass
    # fallback trial list
    trial = (prefer or []) + [
        "utf-8",
        "utf-8-sig",
        "utf-16",
        "cp949",   # common for Korean CSVs
        "euc-kr",
        "iso-8859-1",
        "latin-1",
    ]
    tried = set()
    for enc in trial:
        if enc in tried:
            continue
        tried.add(enc)
        try:
            sample_bytes.decode(enc)
            return enc
        except Exception:
            continue
    # last resort: latin-1 always succeeds
    return "latin-1"


def pretty_open_csv(path, max_rows=50, max_col_width=40, encoding="utf-8-sig", use_pandas=True):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    # read raw bytes sample to autodetect encoding (safe for large files)
    with p.open("rb") as fh:
        sample_bytes = fh.read(131072)  # 128KB sample for better detection

    detected = detect_encoding_from_bytes(sample_bytes, prefer=[encoding])
    if detected:
        encoding = detected

    # decode sample for csv.Sniffer
    decoded_sample = sample_bytes.decode(encoding, errors="replace")

    try:
        sniff = csv.Sniffer()
        delim = sniff.sniff(decoded_sample).delimiter
        has_header = sniff.has_header(decoded_sample)
    except Exception:
        delim = ","
        has_header = True

    # try pandas streaming (chunks) if available and requested
    if use_pandas and pd:
        cols = None
        collected = []
        chunk_size = 10000  # tune if needed
        try:
            reader = pd.read_csv(
                p,
                sep=delim,
                encoding=encoding,
                dtype=str,
                keep_default_na=False,
                header=0 if has_header else None,
                iterator=True,
                chunksize=chunk_size,
                low_memory=False,
            )
            for chunk in reader:
                # strip whitespace
                chunk = chunk.applymap(lambda x: x.strip() if isinstance(x, str) else x)
                # drop blank rows in this chunk
                chunk = _drop_blank_rows_df(chunk)
                if chunk.empty:
                    continue
                # set column names when no header
                if not has_header and cols is None:
                    cols = [f"col{i+1}" for i in range(chunk.shape[1])]
                    chunk.columns = cols
                # truncate long cells
                chunk = chunk.applymap(lambda v: _truncate(v, max_col_width))
                # append rows as dicts (memory proportional to max_rows)
                collected.extend(chunk.to_dict(orient="records"))
                if len(collected) >= max_rows:
                    break
        except Exception:
            # fall back to csv module on any pandas-related error
            collected = []
            cols = None

        if collected:
            if cols is None:
                cols = list(pd.DataFrame(collected).columns)
            to_print = pd.DataFrame(collected[:max_rows], columns=cols)
            if tabulate:
                print(tabulate(to_print, headers="keys", tablefmt="github", showindex=False))
            else:
                print(to_print.to_string(index=False))
            return
        # else continue to csv fallback

    # CSV module fallback (streaming, low memory). Use itertools.chain to avoid loading file.
    with p.open("r", encoding=encoding, errors="replace") as fh:
        reader = csv.reader(fh, delimiter=delim)
        try:
            first = next(reader)
        except StopIteration:
            print("CSV is empty")
            return

        if has_header:
            headers = [h.strip() if h is not None else "" for h in first]
            data_iter = reader
        else:
            ncol = len(first)
            headers = [f"col{i+1}" for i in range(ncol)]
            data_iter = itertools.chain([first], reader)

        collected = []
        col_widths = [len(h) for h in headers]
        for row in data_iter:
            # normalize row length, strip and truncate per-cell
            r2 = [(_truncate(x.strip() if x is not None else "", max_col_width)) for x in row]
            if len(r2) < len(headers):
                r2 += [""] * (len(headers) - len(r2))
            # drop entirely empty rows
            if not any(cell.strip() != "" for cell in r2):
                continue
            collected.append(r2)
            for i, cell in enumerate(r2):
                if i >= len(col_widths):
                    col_widths.append(len(cell))
                else:
                    col_widths[i] = max(col_widths[i], len(cell))
            if len(collected) >= max_rows:
                break

    if not collected:
        print("CSV is empty")
        return

    # pretty print header + rows
    def pad(s, w):
        return s + " " * (w - len(s))

    hdr_line = " | ".join(pad(h, col_widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    print(hdr_line)
    print(sep_line)
    for r in collected[:max_rows]:
        if len(r) < len(headers):
            r += [""] * (len(headers) - len(r))
        print(" | ".join(pad(r[i], col_widths[i]) for i in range(len(headers))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretty-print CSV file with nice columns and spacing.")
    parser.add_argument("csvfile", help="path to CSV file")
    parser.add_argument("--rows", type=int, default=50, help="max number of rows to display")
    parser.add_argument("--maxcol", type=int, default=40, help="max column width (truncates long cells)")
    parser.add_argument("--no-pandas", action="store_true", help="force using csv module fallback (avoid pandas)")
    args = parser.parse_args()

    try:
        pretty_open_csv(
            args.csvfile,
            max_rows=args.rows,
            max_col_width=args.maxcol,
            use_pandas=(not args.no_pandas),
        )
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(1)