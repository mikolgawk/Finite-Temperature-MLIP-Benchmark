"""Resume and checkpoint the per-trajectory results shared by v2 RMSE runners."""

import csv
import json
from pathlib import Path


def trajectory_key(path):
    """Match saved trajectories even when the reference data root has moved."""
    path = Path(path)
    return path.parent.name, path.name


class RmseResults:
    def __init__(self, output, files, *, force=False):
        self.output = output
        self.failure_file = output.with_suffix('.failures.json')
        self.rows = []
        previous_failures = []
        if not force:
            if output.exists():
                with output.open(newline='') as handle:
                    reader = csv.DictReader(handle)
                    if not reader.fieldnames or 'trajectory' not in reader.fieldnames:
                        raise ValueError(f'Missing trajectory column in {output}; use --force to recompute.')
                    self.rows = list(reader)
            if self.failure_file.exists():
                previous_failures = json.loads(self.failure_file.read_text())
                if not isinstance(previous_failures, list) or any(
                    not isinstance(failure, dict) or not failure.get('file')
                    for failure in previous_failures
                ):
                    raise ValueError(f'Expected a list of failures with file paths in {self.failure_file}')

        if force or not output.exists():
            self.files = list(files)
        elif self.failure_file.exists():
            failed = {trajectory_key(failure['file']) for failure in previous_failures}
            self.files = [path for path in files if trajectory_key(path) in failed]
        else:
            completed = {trajectory_key(row['trajectory']) for row in self.rows}
            self.files = [path for path in files if trajectory_key(path) not in completed]

        scheduled = {trajectory_key(path) for path in self.files}
        # Keep failures whose reference files are missing or currently ineligible.
        self.failures = [failure for failure in previous_failures
                         if trajectory_key(failure['file']) not in scheduled]
        # Checkpoints must also remember trajectories not yet attempted if interrupted.
        self.failures.extend({'file': str(path), 'error': 'Evaluation not completed'}
                             for path in self.files)

    def record(self, path, row, failures):
        """Merge one attempt, replacing its partial row and old failure records."""
        key = trajectory_key(path)
        self.failures = [failure for failure in self.failures
                         if trajectory_key(failure['file']) != key]
        self.failures.extend(failures)
        if row is not None:
            self.rows = [previous for previous in self.rows
                         if trajectory_key(previous['trajectory']) != key]
            self.rows.append(row)

        # Keep this trajectory retryable until its new row is committed, even if
        # it succeeded and the save is interrupted while other failures remain.
        checkpoint_failures = self.failures
        if not any(trajectory_key(failure['file']) == key for failure in self.failures):
            checkpoint_failures = [*self.failures,
                                   {'file': str(path), 'error': 'Result checkpoint not completed'}]
        self._write_failures(checkpoint_failures)
        if self.rows:
            temporary = self.output.with_suffix('.csv.tmp')
            with temporary.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
            temporary.replace(self.output)
        else:
            self.output.unlink(missing_ok=True)
        self._write_failures(self.failures)

    def _write_failures(self, failures):
        if failures:
            temporary = self.failure_file.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(failures, indent=2) + '\n')
            temporary.replace(self.failure_file)
        else:
            self.failure_file.unlink(missing_ok=True)

    def finish(self):
        if self.files and self.output.exists():
            print(f'Saved {self.output}')
        if self.failures:
            raise SystemExit(f'Evaluation incomplete: {len(self.failures)} failures; see {self.failure_file}')
