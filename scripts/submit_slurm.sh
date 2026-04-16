#!/bin/bash
# submit_slurm.sh — Submit TSMM benchmark job to BSCC Slurm cluster
#
# Usage:
#   bash scripts/submit_slurm.sh              # required problems only
#   bash scripts/submit_slurm.sh --all        # required + optional
#   bash scripts/submit_slurm.sh --dry-run    # print job script, don't submit
#
# The job binds to NUMA node 0 (CPUs 0-23, 24 cores) via numactl.
# After the job finishes, copy web/results.json to your local machine
# and open the web dashboard to view results.

set -euo pipefail
cd "$(dirname "$0")/.."

# ── Config ───────────────────────────────────────────────────
PARTITION="${SLURM_PARTITION:-cpu}"
NODES=1
TASKS=1
CPUS_PER_TASK=24         # One NUMA node on Xeon Platinum 9242
NUMA_NODE=0              # numactl bind to node 0 (CPUs 0-23)
TIME_LIMIT="02:00:00"
MEM="64G"
BLAS="${BLAS:-mkl}"
EXTRA_BENCH_ARGS=""

# Parse args
ALL_PROBLEMS=false
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --all)     ALL_PROBLEMS=true ;;
        --dry-run) DRY_RUN=true ;;
        *)         EXTRA_BENCH_ARGS="$EXTRA_BENCH_ARGS $arg" ;;
    esac
done

if [ "$ALL_PROBLEMS" = false ]; then
    EXTRA_BENCH_ARGS="--required-only $EXTRA_BENCH_ARGS"
fi

# ── Job script ───────────────────────────────────────────────
JOBSCRIPT=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=tsmm_bench
#SBATCH --partition=$PARTITION
#SBATCH --nodes=$NODES
#SBATCH --ntasks=$TASKS
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --time=$TIME_LIMIT
#SBATCH --mem=$MEM
#SBATCH --output=logs/tsmm_%j.out
#SBATCH --error=logs/tsmm_%j.err

set -euo pipefail
cd "\$SLURM_SUBMIT_DIR"
mkdir -p logs web

# Load required modules (adjust to your cluster's module system)
module load intel/2020 mkl/2020 gcc/10 || true
# Alternatively: module load compiler/intel mpi/intel mkl

# Build
make BLAS=$BLAS -j$CPUS_PER_TASK

# Set thread count to one NUMA node's core count
export OMP_NUM_THREADS=$CPUS_PER_TASK
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# For MKL: limit its internal threading to match
export MKL_NUM_THREADS=$CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$CPUS_PER_TASK

echo "=== Job info ==="
echo "SLURM_JOB_ID: \$SLURM_JOB_ID"
echo "SLURM_NODELIST: \$SLURM_NODELIST"
echo "OMP_NUM_THREADS: \$OMP_NUM_THREADS"
lscpu | grep -E 'Model name|Socket|Core|NUMA|MHz'

# Run benchmark bound to NUMA node $NUMA_NODE
numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE \\
    ./benchmark --output web/results.json $EXTRA_BENCH_ARGS

echo "=== Benchmark complete ==="
echo "Results: web/results.json"
EOF
)

echo "=== Slurm Job Script ==="
echo "$JOBSCRIPT"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "(dry-run: not submitting)"
    exit 0
fi

# ── Submit ───────────────────────────────────────────────────
mkdir -p logs
TMP_SCRIPT=$(mktemp /tmp/tsmm_job_XXXXX.sh)
echo "$JOBSCRIPT" > "$TMP_SCRIPT"
JOB_ID=$(sbatch "$TMP_SCRIPT" | awk '{print $NF}')
rm "$TMP_SCRIPT"

echo "Submitted job $JOB_ID"
echo ""
echo "Monitor:  squeue -j $JOB_ID"
echo "Log:      tail -f logs/tsmm_${JOB_ID}.out"
echo ""
echo "After completion:"
echo "  scp <cluster>:\$(pwd)/web/results.json web/results.json"
echo "  make web   # open dashboard"
