"""
Process_GIS_Files.py

Combined utilities for reading GIS shapefiles and combining location CSVs.

Provides two classes:
 - ReadGISFile: read shapefile layers, export attribute tables to CSV, and optional plotting
 - CombineLocation: combine city/district/area CSVs into nested JSON/flat CSV
"""

from __future__ import annotations
import os
import sys
import csv
import json
# argparse removed: script now runs tasks automatically when executed
import re
import glob
from typing import List, Dict, Tuple, Optional, Any
import shutil

try:
    import geopandas as gpd
    import matplotlib.pyplot as plt
except Exception:
    # geopandas/plt are optional for some functionality; surface clear error when used
    gpd = None  # type: ignore
    plt = None  # type: ignore

try:
    import pandas as pd  # noqa: F401
except Exception:
    # pandas is not strictly required for all operations here
    pd = None  # type: ignore


class ReadGISFile:
    """Read a shapefile and export attributes to CSV, with optional plotting."""

    def __init__(self, shp_path: Optional[str] = None, csv_out: Optional[str] = None, plot: bool = False):
        self.shp_path = shp_path
        self.csv_out = csv_out
        self.plot = plot

    def read_shapefile(self, shp_path: str) -> Any:
        if gpd is None:
            raise RuntimeError("geopandas is required to read shapefiles. Please install it in your environment.")
        if not os.path.exists(shp_path):
            raise FileNotFoundError(f"Shapefile not found: {shp_path}")
        gdf = gpd.read_file(shp_path)
        return gdf

    def export_attributes(self, gdf: Any, csv_path: Optional[str]) -> None:
        # Convert geometry to WKT so it can be saved in tabular formats
        df = gdf.copy()
        if 'geometry' in df.columns:
            df["geometry_wkt"] = df.geometry.to_wkt()
            df_no_geom = df.drop(columns=["geometry"])
        else:
            df_no_geom = df

        if csv_path:
            os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
            df_no_geom.to_csv(csv_path, index=False)
            print(f"✓ Saved CSV:   {csv_path}")

    def plot_gdf(self, gdf: Any, title: str) -> None:
        if plt is None:
            raise RuntimeError("matplotlib is required to plot. Please install matplotlib.")
        ax = gdf.plot(edgecolor="black", facecolor="lightcoral", figsize=(10, 8))
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Longitude / X")
        ax.set_ylabel("Latitude / Y")
        plt.tight_layout()
        plt.show()


class CombineLocation:
    """Combine three administrative-level CSVs (city, district, area) into nested JSON/CSV."""

    @staticmethod
    def read_csv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
        try:
            csv.field_size_limit(sys.maxsize)
        except OverflowError:
            csv.field_size_limit(2**31 - 1)

        with open(path, newline='', errors='replace') as f:
            reader = csv.DictReader(f)
            rows = [{k: (v.strip() if v is not None else '') for k, v in row.items()} for row in reader]
            headers = reader.fieldnames or []
        return headers, rows

    @staticmethod
    def detect_code_column(headers: List[str], rows: List[Dict[str, str]]) -> Optional[str]:
        patterns = [r'code', r'cd\b', r'adm', r'코드', r'ADM']
        for h in headers:
            for p in patterns:
                if re.search(p, h, re.I):
                    return h
        numeric_scores = {}
        for h in headers:
            cnt = 0
            total = 0
            for r in rows:
                v = r.get(h, '')
                if not v:
                    continue
                total += 1
                if re.fullmatch(r"\d+", v):
                    cnt += 1
            if total > 0:
                numeric_scores[h] = cnt / total
        if numeric_scores:
            best, score = max(numeric_scores.items(), key=lambda kv: kv[1])
            if score > 0.5:
                return best
        return headers[0] if headers else None

    @staticmethod
    def detect_name_column(headers: List[str], rows: List[Dict[str, str]]) -> Optional[str]:
        patterns = [r'name', r'nm\b', r'NAME', r'NM', r'명', r'이름', r'명칭']
        for h in headers:
            for p in patterns:
                if re.search(p, h, re.I):
                    return h
        for h in headers:
            if not re.search(r'code|cd|adm', h, re.I):
                return h
        return headers[0] if headers else None

    @staticmethod
    def detect_code_lengths(rows: List[Dict[str, str]], code_col: str) -> List[int]:
        lengths = sorted({len(r.get(code_col, '')) for r in rows if r.get(code_col)})
        return [l for l in lengths if l > 0]

    @staticmethod
    def build_mapping(rows: List[Dict[str, str]], code_col: str, name_col: Optional[str]) -> Dict[str, Dict[str, str]]:
        mapping: Dict[str, Dict[str, str]] = {}
        for r in rows:
            code = r.get(code_col, '').strip()
            name = r.get(name_col, '').strip() if name_col else ''
            if not code:
                continue
            mapping[code] = {'code': code, 'name': name}
        return mapping

    @classmethod
    def combine_locations(
        cls,
        city_file: str,
        district_file: str,
        area_file: str,
        c_code_hint: Optional[str] = None,
        d_code_hint: Optional[str] = None,
        a_code_hint: Optional[str] = None,
        c_name_hint: Optional[str] = None,
        d_name_hint: Optional[str] = None,
        a_name_hint: Optional[str] = None,
    ) -> List[Dict]:
        c_h, c_rows = cls.read_csv(city_file)
        d_h, d_rows = cls.read_csv(district_file)
        a_h, a_rows = cls.read_csv(area_file)

        c_code = c_code_hint or cls.detect_code_column(c_h, c_rows)
        d_code = d_code_hint or cls.detect_code_column(d_h, d_rows)
        a_code = a_code_hint or cls.detect_code_column(a_h, a_rows)

        c_name = c_name_hint or cls.detect_name_column(c_h, c_rows)
        d_name = d_name_hint or cls.detect_name_column(d_h, d_rows)
        a_name = a_name_hint or cls.detect_name_column(a_h, a_rows)

        if not (c_code and d_code and a_code):
            raise ValueError('Unable to detect code columns for all three files; consider passing explicit column names via CLI')

        cities = cls.build_mapping(c_rows, c_code, c_name)
        districts = cls.build_mapping(d_rows, d_code, d_name)
        areas = cls.build_mapping(a_rows, a_code, a_name)

        c_lens = cls.detect_code_lengths(c_rows, c_code)
        d_lens = cls.detect_code_lengths(d_rows, d_code)
        a_lens = cls.detect_code_lengths(a_rows, a_code)

        c_pref = c_lens[0] if c_lens else 0
        d_pref = d_lens[0] if d_lens else 0
        a_pref = a_lens[0] if a_lens else 0

        city_list: List[Dict] = []
        district_areas: Dict[str, List[Dict]] = {dcode: [] for dcode in districts}
        for acode, ainfo in areas.items():
            parent = None
            if d_pref and len(acode) >= d_pref:
                prefix = acode[:d_pref]
                if prefix in districts:
                    parent = prefix
            if parent is None:
                for dcode in districts:
                    if acode.startswith(dcode):
                        parent = dcode
                        break
            if parent:
                district_areas.setdefault(parent, []).append({'code': acode, 'name': ainfo.get('name', '')})
            else:
                district_areas.setdefault('__orphan_areas__', []).append({'code': acode, 'name': ainfo.get('name', '')})

        for ccode, cinfo in cities.items():
            city_obj = {'code': ccode, 'name': cinfo.get('name', ''), 'districts': []}
            for dcode, dinfo in districts.items():
                belongs = False
                if c_pref and len(dcode) >= c_pref and dcode[:c_pref] == ccode[:c_pref]:
                    belongs = True
                elif dcode.startswith(ccode):
                    belongs = True
                if belongs:
                    d_obj = {'code': dcode, 'name': dinfo.get('name', ''), 'areas': district_areas.get(dcode, [])}
                    city_obj['districts'].append(d_obj)
            city_list.append(city_obj)

        assigned_districts = {d['code'] for c in city_list for d in c['districts']}
        unassigned = [dcode for dcode in districts.keys() if dcode not in assigned_districts]
        if unassigned:
            orphan_city = {'code': '__orphan_districts__', 'name': 'Orphan districts', 'districts': []}
            for dcode in unassigned:
                dinfo = districts[dcode]
                orphan_city['districts'].append({'code': dcode, 'name': dinfo.get('name',''), 'areas': district_areas.get(dcode, [])})
            city_list.append(orphan_city)

        if district_areas.get('__orphan_areas__'):
            orphan_city = {'code': '__orphan_areas__', 'name': 'Orphan areas', 'districts': [{'code': '__orphan__', 'name': 'Unassigned', 'areas': district_areas.get('__orphan_areas__')}]} 
            city_list.append(orphan_city)

        return city_list

    @staticmethod
    def write_json(data: List[Dict], out_path: str):
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def write_flat_csv(data: List[Dict], out_path: str):
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['city_code','city_name','district_code','district_name','area_code','area_name'])
            for city in data:
                for district in city.get('districts', []):
                    if not district.get('areas'):
                        writer.writerow([city.get('code'), city.get('name'), district.get('code'), district.get('name'), '', ''])
                    else:
                        for area in district.get('areas', []):
                            writer.writerow([city.get('code'), city.get('name'), district.get('code'), district.get('name'), area.get('code'), area.get('name')])


# The script runs both tasks automatically when executed.
# It uses the same defaults as the original scripts in the repo.


def main() -> int:
    # create output folder for generated files
    out_dir = "Location_Data"
    os.makedirs(out_dir, exist_ok=True)

    # 1) Ensure CSVs for all three administrative levels exist in out_dir. If missing, try to find
    #    shapefiles in Dataset/** and export attributes to CSV in out_dir.
    levels = [
        ("bnd_sido_00_2024_2Q.csv", "sido"),
        ("bnd_sigungu_00_2024_2Q.csv", "sigungu"),
        ("bnd_dong_00_2024_2Q.csv", "dong"),
    ]

    for csv_name, keyword in levels:
        out_path = os.path.join(out_dir, csv_name)
        # if file already in out_dir, skip
        if os.path.exists(out_path):
            print(f"CSV exists in {out_dir}: {csv_name} — skipping creation")
            continue

        # if exists in repo root, move it into out_dir
        if os.path.exists(csv_name):
            try:
                print(f"Moving existing {csv_name} into {out_dir}/")
                shutil.move(csv_name, out_path)
                continue
            except Exception as e:
                print(f"Failed to move {csv_name} into {out_dir}: {e}")

        print(f"CSV missing: {csv_name}. Searching for a matching shapefile in Dataset/...")
        pattern = os.path.join("Dataset", "**", f"*{keyword}*.shp")
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            print(f"No shapefile matching '*{keyword}*.shp' found under Dataset/. Cannot create {csv_name}.")
            continue

        shp_path = matches[0]
        print(f"Found shapefile for {keyword}: {shp_path}")
        reader = ReadGISFile(shp_path=shp_path, csv_out=out_path, plot=False)
        try:
            gdf = reader.read_shapefile(shp_path)
            # export to the target CSV filename in out_dir
            reader.export_attributes(gdf, out_path)
        except Exception as e:
            print(f"Failed to create {csv_name} from {shp_path}: {e}")

    # 2) Combine location CSVs into nested JSON and flattened CSV
    print("\n--- Combining location CSVs into JSON/CSV")
    try:
        city_csv = os.path.join(out_dir, 'bnd_sido_00_2024_2Q.csv')
        district_csv = os.path.join(out_dir, 'bnd_sigungu_00_2024_2Q.csv')
        area_csv = os.path.join(out_dir, 'bnd_dong_00_2024_2Q.csv')

        combined = CombineLocation.combine_locations(
            city_csv,
            district_csv,
            area_csv,
            c_code_hint='SIDO_CD',
            c_name_hint='SIDO_NM',
            d_code_hint='SIGUNGU_CD',
            d_name_hint='SIGUNGU_NM',
            a_code_hint='ADM_CD',
            a_name_hint='ADM_NM',
        )
        out_json = os.path.join(out_dir, 'combined_locations.json')
        out_flat_csv = os.path.join(out_dir, 'combined_locations.csv')
        CombineLocation.write_json(combined, out_json)
        print(f'Wrote nested JSON to {out_json}')
        CombineLocation.write_flat_csv(combined, out_flat_csv)
        print(f'Wrote flattened CSV to {out_flat_csv}')
    except Exception as e:
        print(f"Error combining locations: {e}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
