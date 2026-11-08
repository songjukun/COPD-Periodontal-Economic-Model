"""Run the fixed-seed analysis and compare numerical outputs with saved references."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / '06_Analysis'


def main():
    commands = [
        ['analysis.py', '--n', '50000', '--seed', '20261015'],
        ['supplemental_analysis.py'], ['zhou2014_analysis.py'],
        ['threshold_analysis.py'], ['maintenance_schedule_analysis.py'],
        ['maintenance_roots.py'], ['plot_revision_figures.py']]
    for command in commands:
        subprocess.run([sys.executable, str(ANALYSIS / command[0]), *command[1:]],
                       cwd=ROOT, check=True)
    # Preserve the original plotting filenames, and provide manuscript numbering.
    fig = ROOT / '04_Figures'
    mapping = {'Figure1_CEAC': 'Figure2_CEAC',
               'Figure2_CostEffectivenessPlane': 'Figure3_CostEffectivenessPlane',
               'Figure3_Independent_Thresholds': 'Figure4_Independent_Thresholds'}
    for old, new in mapping.items():
        for ext, folder in [('png', 'PNG'), ('tiff', '300dpi_TIFF'), ('pdf', 'Vector_PDF')]:
            shutil.copy2(fig / f'{old}.{ext}', fig / folder / f'{new}.{ext}')
    checked = []
    for reference in sorted((ROOT / 'reference_results').glob('*.csv')):
        actual = pd.read_csv(ANALYSIS / 'results' / reference.name)
        expected = pd.read_csv(reference)
        pd.testing.assert_frame_equal(actual, expected, check_exact=False,
                                      rtol=1e-9, atol=1e-10)
        checked.append(reference.name)
    name = 'maintenance_roots_20261020.json'
    actual = pd.DataFrame(json.loads((ANALYSIS / 'results' / name).read_text(encoding='utf-8')))
    expected = pd.DataFrame(json.loads((ROOT / 'reference_results' / name).read_text(encoding='utf-8')))
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-9, atol=1e-10)
    checked.append(name)
    report = {'status': 'passed', 'reference_files_compared': checked,
              'note': 'Numerical reproducibility does not establish clinical validity.'}
    (ANALYSIS / 'results' / 'reference_comparison.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
