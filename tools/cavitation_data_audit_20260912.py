"""Read-only source audit and bounded nested-archive staging; never execute source code."""
import argparse
import collections
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/cavitation_20260912/data_audit'
SOURCE = ROOT / "data/raw/Ali's Project.zip"
SEVEN = pathlib.Path('C:/Program Files/7-Zip/7z.exe')

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def force_audit():
    import numpy as np
    from scipy.signal import welch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = []
    fig, axs = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    for row, case in enumerate(['Case 13', 'Case 14', 'Case 16']):
        for col, force in enumerate(['cl','cd']):
            source = OUT/'representative'/case/(force+'-1-history.out')
            values = np.loadtxt(source, skiprows=3)
            assert values.shape[1] == 3 and np.all(np.isfinite(values))
            steps, times, signal = values.T
            differences = np.diff(times)
            assert np.all(np.diff(steps)==1) and np.all(differences>0)
            dt = float(np.median(differences))
            assert np.max(np.abs(differences-dt)) < 1e-10
            half = times >= (times[0]+times[-1])/2
            early = times <= 2.12
            def stats(mask):
                selected = signal[mask]
                return {'count': int(mask.sum()), 'mean': float(selected.mean()), 'standard_deviation': float(selected.std(ddof=1)), 'min': float(selected.min()), 'max': float(selected.max())}
            frequency, power = welch(signal[half], fs=1/dt, nperseg=min(4096,int(half.sum())), detrend='linear')
            positive = frequency > 0
            peak = int(np.argmax(power[positive]))
            record = {'case':case,'quantity':force,'source':str(source.relative_to(ROOT)),'sha256':sha256(source),'rows':len(values),'first_time':float(times[0]),'last_time':float(times[-1]),'dt':dt,'finite':True,'steps_contiguous':True,'first_2p12_seconds':stats(early),'last_half':stats(half),'last_half_welch_peak_hz':float(frequency[positive][peak]),'welch_frequency_bin_hz':float(frequency[1]),'welch_nperseg':min(4096,int(half.sum())),'caveat':'Descriptive linear-detrended Welch maximum, not validated shedding frequency or Strouhal; no stationarity or grid/time convergence established.'}
            records.append(record)
            ax=axs[row,col]
            ax.plot(times,signal,lw=.55)
            ax.axvspan(0.08,2.12,color='orange',alpha=.2,label='saved spatial-field span')
            ax.axvspan(times[half][0],times[-1],color='green',alpha=.09,label='last-half statistics')
            ax.set_title(case+' '+force.upper())
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(force.upper())
            if row==0 and col==0:
                ax.legend(fontsize=8)
    fig.savefig(OUT/'FORCE_HISTORY_REVIEW.png',dpi=150)
    plt.close(fig)
    (OUT/'FORCE_HISTORY_AUDIT.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(json.dumps(records,indent=2))

def review():
    import h5py
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    report = []
    fig, axs = plt.subplots(3, 3, figsize=(15, 8), constrained_layout=True)
    for row, case in enumerate(['Case 13', 'Case 14', 'Case 16']):
        stem = case.replace(' ', '_')
        archive = np.load(OUT / (stem+'_raster.npz'))
        provenance = json.loads((OUT / (stem+'_raster.json')).read_text())
        record = {'case': case, 'split': provenance['split'], 'raster_sha256': sha256(OUT / (stem+'_raster.npz')), 'porous_centroids_inside_solid': provenance['porous_centroids_inside_solid'], 'porous_raster_cells': provenance['porous_raster_cells'], 'time_range': archive['times'][[0,-1]].tolist(), 'frames': len(archive['times'])}
        first = next((OUT / 'representative' / case).glob('*.cas.h5'))
        with h5py.File(first, 'r') as file:
            settings = file['settings/Rampant Variables'][0].decode()
            evidence = re.search(r'\(mass-transfer \(2 3 \(cavitation .{0,550}', settings).group(0)
            record['active_mass_transfer_settings_excerpt'] = evidence
            record['legacy_vapor_pressure_default'] = re.search(r'\(mp/cvt/vapor-p [^\)]+\)', settings).group(0)
        p_audits = []
        for path in sorted((OUT / 'representative' / case).glob('*.dat.h5')):
            with h5py.File(path, 'r') as file:
                variables = file['settings/Data Variables'][0].decode()
                op = float(re.search(r'\(operating-pressure ([^\)]+)\)', variables).group(1))
                p = file['results/1/phase-1/cells/SV_P/1'][:]
                pv = file['results/1/phase-1/cells/SV_MT_VAPOR_PRESS/1'][:]
                alpha = file['results/1/phase-3/cells/SV_VOF/1'][:]
                p_audits.append({'file': path.name, 'operating_pressure': op, 'actual_saved_vapor_pressure_minmax': [float(pv.min()),float(pv.max())], 'absolute_pressure_minmax': [float(p.min()+op),float(p.max()+op)], 'vapor_gt_half_cells': int((alpha>.5).sum()), 'vapor_gt_half_below_actual_pvap_fraction': float(np.mean((p+op < pv)[alpha>.5])) if np.any(alpha>.5) else None})
        record['pressure_audit'] = p_audits
        record['interpretation'] = 'Use active mass-transfer p-vap and saved SV_MT_VAPOR_PRESS, not legacy mp/cvt/vapor-p. Elevated configured vapor pressure is not a validated physical saturation pressure. Pressure audit does not establish CFD convergence.'
        report.append(record)
        for col, index in enumerate([0,8,17]):
            ax = axs[row,col]
            alpha = np.ma.masked_where(archive['wall'], archive['alpha_v'][index])
            ax.pcolormesh(archive['x'], archive['y'], alpha, shading='nearest', vmin=0,vmax=1,cmap='viridis',rasterized=True)
            ax.add_collection(LineCollection(archive['wall_segments'], colors='black', linewidths=.6))
            ax.set_facecolor('gray')
            ax.set_aspect('equal')
            ax.set_title(f'{case}, t={archive["times"][index]:.2f} s')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
    fig.suptitle('Native vapor fraction mapped by nearest cell centroid; gray = exact wall-segment interior\nCase 13 train / 14 validation / 16 test; displayed frames 0, 8, 17; fixed alpha scale 0–1')
    fig.savefig(OUT/'RASTER_REVIEW.png', dpi=150)
    plt.close(fig)
    (OUT/'DATA_REVIEW.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps([{key:value for key,value in record.items() if key not in ['pressure_audit','active_mass_transfer_settings_excerpt']} for record in report],indent=2))

def rasterize(case):
    import h5py
    import numpy as np
    from scipy.spatial import cKDTree
    case_dir = OUT / 'representative' / case
    casefiles = sorted(case_dir.rglob('*.cas.h5'))
    datafiles = sorted(case_dir.rglob('*.dat.h5'))
    assert len(casefiles) == len(datafiles) and len(casefiles) > 0
    provenance = {'case': case, 'files': [], 'raster_method': 'nearest vertex-mean cell centroid; native cell fields retained; not conservative', 'roi': [-0.03, 0.47, -0.07, 0.07], 'width': 384, 'height': 192, 'split': {'Case 13': 'train', 'Case 14': 'validation', 'Case 16': 'test', 'Case1LES': 'train', 'Case 19': 'validation', 'Case 24': 'test', 'Case 23': 'test'}[case]}
    with h5py.File(casefiles[0], 'r') as meshfile:
        mesh = meshfile['meshes/1']
        assert int(mesh.attrs['dimension'][0]) == 2, 'Only 2-D native meshes supported'
        coords = mesh['nodes/coords/1'][:]
        assert np.all(mesh['faces/nodes/1/nnodes'][:] == 2)
        edges = mesh['faces/nodes/1/nodes'][:].reshape(-1, 2).astype(np.int64) - 1
        ncell = int(mesh.attrs['cellCount'][0])
        centroids = np.zeros((ncell, 2))
        counts = np.zeros(ncell)
        midpoint = coords[edges].mean(1)
        for side in ['c0', 'c1']:
            for ds in mesh[f'faces/{side}'].values():
                start = int(ds.attrs['minId'][0]) - 1
                cellids = ds[:].astype(np.int64) - 1
                assert np.all((cellids >= 0) & (cellids < ncell))
                np.add.at(centroids, cellids, midpoint[start:start+len(ds)])
                np.add.at(counts, cellids, 1)
        assert np.all(counts >= 3)
        centroids /= counts[:, None]
        topology = mesh['faces/zoneTopology']
        names = topology['name'][0].decode().split(';')
        wall_edges = np.concatenate([edges[int(lo)-1:int(hi)] for name, lo, hi in zip(names, topology['minId'][:], topology['maxId'][:]) if name.startswith('airfoil')])
        x = np.linspace(-0.03, 0.47, 384)
        y = np.linspace(-0.07, 0.07, 192)
        xx, yy = np.meshgrid(x, y)
        query = np.column_stack([xx.ravel(), yy.ravel()])
        distance, indices = cKDTree(centroids).query(query)
        inside = np.zeros(query.shape[0], dtype=bool)
        for a, b in coords[wall_edges]:
            cross = (a[1] > query[:, 1]) != (b[1] > query[:, 1])
            if a[1] != b[1]:
                inside ^= cross & (query[:, 0] < (b[0]-a[0]) * (query[:, 1]-a[1]) / (b[1]-a[1]) + a[0])
        wall = inside.reshape(xx.shape)
        cell_topology = mesh['cells/zoneTopology']
        porous_native = np.zeros(ncell, dtype=bool)
        for name, low, high in zip(cell_topology['name'][0].decode().split(';'), cell_topology['minId'][:], cell_topology['maxId'][:]):
            if name == 'porous':
                porous_native[int(low)-1:int(high)] = True
        porous_centroids = centroids[porous_native]
        porous_inside_solid = np.zeros(len(porous_centroids), dtype=bool)
        for a, b in coords[wall_edges]:
            if a[1] != b[1]:
                cross = (a[1] > porous_centroids[:, 1]) != (b[1] > porous_centroids[:, 1])
                porous_inside_solid ^= cross & (porous_centroids[:, 0] < (b[0]-a[0]) * (porous_centroids[:, 1]-a[1]) / (b[1]-a[1]) + a[0])
        assert not porous_inside_solid.any(), 'Porous fluid must not be classified solid'
        porous_mask = porous_native[indices].reshape(xx.shape) & ~wall
        settings = meshfile['settings/Rampant Variables'][0].decode()
        assert '(3 phase-domain vapor)' in settings and '(material . water-vapor)' in settings
        provenance['phase_evidence'] = 'Rampant Variables: (3 phase-domain vapor), material water-vapor'
        provenance['n_cells'] = ncell
        provenance['porous_native_cells'] = int(porous_native.sum())
        provenance['porous_centroids_inside_solid'] = int(porous_inside_solid.sum())
        provenance['porous_raster_cells'] = int(porous_mask.sum())
        provenance['mesh_first_sha256'] = sha256(casefiles[0])
        provenance['wall_edges'] = len(wall_edges)
        provenance['wall_edge_degree_counts'] = dict(zip(*[a.tolist() for a in np.unique(np.bincount(wall_edges.ravel())[np.bincount(wall_edges.ravel()) > 0], return_counts=True)]))
        provenance['raster_wall_cells'] = int(wall.sum())
        provenance['centroid_face_counts'] = dict(zip(*[a.tolist() for a in np.unique(counts, return_counts=True)]))
        mesh_hash = hashlib.sha256(coords.tobytes()+edges.tobytes()).hexdigest()
    arrays = {name: [] for name in ['alpha_v', 'pressure', 'u', 'v', 'wall_distance', 'mass_transfer']}
    times = []
    native_stats = []
    for i, (casefile, datafile) in enumerate(zip(casefiles, datafiles)):
        assert str(casefile).replace('.cas.', '.dat.') == str(datafile)
        with h5py.File(casefile, 'r') as other:
            other_coords = other['meshes/1/nodes/coords/1'][:]
            other_edges = other['meshes/1/faces/nodes/1/nodes'][:].reshape(-1, 2).astype(np.int64)-1
            assert hashlib.sha256(other_coords.tobytes()+other_edges.tobytes()).hexdigest() == mesh_hash
        with h5py.File(datafile, 'r') as data:
            fields = {'alpha_v': 'phase-3/cells/SV_VOF/1', 'pressure': 'phase-1/cells/SV_P/1', 'u': 'phase-1/cells/SV_U/1', 'v': 'phase-1/cells/SV_V/1', 'wall_distance': 'phase-1/cells/SV_WALL_DIST/1', 'mass_transfer': 'phase-1/cells/SV_MASS_TRANSFER/1'}
            variables = data['settings/Data Variables'][0].decode()
            time = float(re.search(r'\(flow-time ([^\)]+)\)', variables).group(1))
            op = float(re.search(r'\(operating-pressure ([^\)]+)\)', variables).group(1))
            times.append(time)
            stat = {'time': time, 'operating_pressure': op}
            for name, path in fields.items():
                values = data['results/1/'+path][:]
                assert values.shape == (ncell,) and np.all(np.isfinite(values))
                if name == 'alpha_v':
                    assert values.min() >= -1e-8 and values.max() <= 1+1e-8
                stat[name] = dict(min=float(values.min()), max=float(values.max()), mean=float(values.mean()))
                arrays[name].append(values[indices].reshape(xx.shape).astype(np.float32))
            liquid = data['results/1/phase-2/cells/SV_VOF/1'][:]
            vapor = data['results/1/phase-3/cells/SV_VOF/1'][:]
            stat['max_phase_sum_error'] = float(np.max(np.abs(liquid+vapor-1)))
            native_stats.append(stat)
        provenance['files'].append({'case_path': str(casefile.relative_to(ROOT)), 'data_path': str(datafile.relative_to(ROOT)), 'data_sha256': sha256(datafile), 'time': time})
        print(f'{case} frame {i+1}/{len(datafiles)} t={time}', flush=True)
    assert np.all(np.diff(times) > 0)
    destination = OUT / (case.replace(' ', '_') + '_raster.npz')
    np.savez_compressed(destination, **{key: np.stack(value) for key, value in arrays.items()}, times=np.asarray(times), x=x, y=y, wall=wall, geometry_mask=wall, porous_mask=porous_mask, nearest_distance=distance.reshape(xx.shape), source_cell_id=(indices+1).reshape(xx.shape), wall_segments=coords[wall_edges])
    provenance['native_stats'] = native_stats
    provenance['raster_sha256'] = sha256(destination)
    provenance['temporal_dt'] = np.diff(times).tolist()
    (OUT / (case.replace(' ', '_') + '_raster.json')).write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(str(destination), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', default='Case 13.rar')
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('--rasterize', action='store_true')
    parser.add_argument('--review', action='store_true')
    parser.add_argument('--forces', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.forces:
        force_audit()
        return
    if args.review:
        review()
        return
    if args.rasterize:
        rasterize(args.case.replace('.rar', ''))
        return
    with zipfile.ZipFile(SOURCE) as archive:
        entries = archive.infolist()
        records = [dict(path=e.filename, bytes=e.file_size, compressed_bytes=e.compress_size, crc32=f'{e.CRC:08x}') for e in entries]
        audit = dict(source=str(SOURCE), source_bytes=SOURCE.stat().st_size, count=len(entries), uncompressed_bytes=sum(e.file_size for e in entries), entries=records)
        (OUT / 'outer_inventory.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
        selected = [e for e in entries if pathlib.PurePosixPath(e.filename).name == args.case]
        if len(selected) != 1:
            raise ValueError('Expected exactly one case')
        entry = selected[0]
        if entry.file_size > 1_100_000_000:
            raise ValueError('Nested archive exceeds 1.1 GB staging cap')
        if shutil.disk_usage(OUT).free < entry.file_size * 2 + 2_000_000_000:
            raise ValueError('Insufficient space')
        destination = OUT / args.case
        if not destination.exists():
            with archive.open(entry) as source, destination.open('xb') as target:
                shutil.copyfileobj(source, target, length=1024*1024)
        if destination.stat().st_size != entry.file_size:
            raise ValueError('Staged size mismatch')
    listing = subprocess.run([str(SEVEN), 'l', '-slt', str(destination)], capture_output=True, text=True, check=True).stdout
    (OUT / (destination.stem.replace(' ', '_') + '_listing.txt')).write_text(listing, encoding='utf-8')
    blocks = listing.split('----------\n', 1)[1].strip().split('\n\n')
    members = []
    for block in blocks:
        values = dict(line.split(' = ', 1) for line in block.splitlines() if ' = ' in line)
        if 'Path' not in values:
            continue
        name = pathlib.PurePosixPath(values['Path'].replace('\\', '/'))
        if name.is_absolute() or '..' in name.parts or ':' in str(name):
            raise ValueError('Unsafe archive member path')
        if any(values.get(key) for key in ['Symbolic Link', 'Hard Link', 'Copy Link']):
            raise ValueError('Link rejected')
        members.append(values)
    total = sum(int(m.get('Size', 0)) for m in members)
    if total > 1_500_000_000:
        raise ValueError('Expanded case exceeds 1.5 GB cap')
    summary = dict(case=args.case, member_count=len(members), expanded_bytes=total, extensions=dict(collections.Counter(pathlib.PurePosixPath(m['Path']).suffix for m in members)), source_crc32=f'{entry.CRC:08x}')
    (OUT / (destination.stem.replace(' ', '_') + '_inventory.json')).write_text(json.dumps(dict(summary, members=members), indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)
    if args.extract:
        staged = OUT / 'representative'
        result = subprocess.run([str(SEVEN), 'x', str(destination), '-o' + str(staged), '-aos'], capture_output=True, text=True, check=True)
        (OUT / (destination.stem.replace(' ', '_') + '_extraction.txt')).write_text(result.stdout, encoding='utf-8')
        print(result.stdout)

if __name__ == '__main__':
    main()
