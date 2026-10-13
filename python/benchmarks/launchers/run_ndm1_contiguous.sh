#!/usr/bin/env bash
set -u

repo_dir="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$repo_dir"

mode="${1:-pilot}"
root="outputs/contiguous_benchmark"
mkdir -p "$root"

run_case() {
    local name="$1"
    local start_position="$2"
    local gpu_id="$3"
    shift 3
    local freeze_residues=("$@")
    local outdir="$root/$name"
    mkdir -p "$outdir"

    {
        echo "name=$name"
        echo "freeze_residues=${freeze_residues[*]}"
        echo "desired_motif_start_position=$start_position"
        echo "gpu_id=$gpu_id"
        echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "$outdir/run_config.txt"

    CUDA_VISIBLE_DEVICES="$gpu_id" ./.env/bin/python python/energy_gradient.py \
        --steps 10 \
        --prefix-outfile "$name" \
        --output-path "$outdir" \
        --desired-motif-start-position "$start_position" \
        --out-size 8 \
        --freeze-residues "${freeze_residues[@]}" \
        --batch-size 10000 \
        --inpaint \
        --inpaint-dist 0.1 \
        --scaffold-num 100 \
        --pdb-input pdbs/6xbf_nochanges.pdb \
        --weights-load weights/8mer/model_7731_best_weights.pt \
        --silentfile \
        --skip-individual \
        --peptide-chain I \
        --complex-output \
        --score-outputs \
        --gpu \
        > "$outdir/run.log" 2>&1
    local rc=$?
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$outdir/run_config.txt"
    echo "exit_code=$rc" >> "$outdir/run_config.txt"
    return "$rc"
}

run_parallel_batch() {
    local specs=("$@")
    local pids=()
    local names=()
    local failures=0
    local i=0
    for spec in "${specs[@]}"; do
        read -r -a fields <<< "$spec"
        local gpu_id=$((i % 3))
        run_case "${fields[0]}" "${fields[1]}" "$gpu_id" "${fields[@]:2}" &
        pids+=("$!")
        names+=("${fields[0]}")
        i=$((i + 1))
    done
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            echo -e "${names[$i]}\tPASS"
        else
            echo -e "${names[$i]}\tFAIL"
            failures=$((failures + 1))
        fi
    done
    return "$failures"
}

if [[ "$mode" == "pilot" ]]; then
    run_parallel_batch \
        "NDM1_L2_S1004 4 1004 1005" \
        "NDM1_L3_S1003 3 1003 1004 1005" \
        "NDM1_L4_S1002 2 1002 1003 1004 1005"
    pilot_rc=$?
    run_parallel_batch "NDM1_L2_S1008 8 1008 1001"
    wrap_rc=$?
    if (( pilot_rc != 0 || wrap_rc != 0 )); then
        exit 1
    fi
elif [[ "$mode" == "nonwrap" ]]; then
    # The three non-wrap pilots (L2/S1004, L3/S1003, L4/S1002) are already
    # complete. Run the remaining 15 non-wrap windows in five GPU waves.
    failures=0
    run_parallel_batch \
        "NDM1_L2_S1001 1 1001 1002" \
        "NDM1_L2_S1002 2 1002 1003" \
        "NDM1_L2_S1003 3 1003 1004" || failures=$((failures + $?))
    run_parallel_batch \
        "NDM1_L2_S1005 5 1005 1006" \
        "NDM1_L2_S1006 6 1006 1007" \
        "NDM1_L2_S1007 7 1007 1008" || failures=$((failures + $?))
    run_parallel_batch \
        "NDM1_L3_S1001 1 1001 1002 1003" \
        "NDM1_L3_S1002 2 1002 1003 1004" \
        "NDM1_L3_S1004 4 1004 1005 1006" || failures=$((failures + $?))
    run_parallel_batch \
        "NDM1_L3_S1005 5 1005 1006 1007" \
        "NDM1_L3_S1006 6 1006 1007 1008" \
        "NDM1_L4_S1001 1 1001 1002 1003 1004" || failures=$((failures + $?))
    run_parallel_batch \
        "NDM1_L4_S1003 3 1003 1004 1005 1006" \
        "NDM1_L4_S1004 4 1004 1005 1006 1007" \
        "NDM1_L4_S1005 5 1005 1006 1007 1008" || failures=$((failures + $?))
    exit "$failures"
else
    echo "Unknown mode: $mode" >&2
    exit 2
fi
