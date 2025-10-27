# NCCL Debug Log Parser \& Visualizer

Tools to parse and visualize NCCL logs generated with debug mode enabled.

## Requirements

```bash
pip install pandas numpy matplotlib seaborn
```


## Quick Start

### 1. Generate NCCL logs with debug activated

```bash
# Option 1: Full logs (recommended)
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL,P2P <your_command> 2>&1 | tee nccl_debug.log

# Option 2: Only collective operations
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL <your_command> 2>&1 | tee nccl_debug.log

# Example with distributed PyTorch:
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL python -m torch.distributed.launch \
    --nproc_per_node=8 train.py 2>&1 | tee nccl_debug.log
```


### 2. Parse the log and generate analysis

```bash
python nccl_parser.py nccl_debug.log --output results/analysis
```

This will generate multiple CSV files with detailed analysis:

- `analysis_collective_ops_summary.csv` - Summary by operation type
- `analysis_collective_rank_summary.csv` - Summary by rank
- `analysis_collective_rank_op_matrix.csv` - Rank × operation matrix
- `analysis_p2p_comm_count_matrix.csv` - P2P communications matrix (count)
- `analysis_p2p_comm_data_matrix.csv` - P2P communications matrix (data MB)
- `analysis_p2p_rank_summary.csv` - Send/Recv summary by rank
- `analysis_all_collective_ops.csv` - All parsed collective operations
- `analysis_all_p2p_ops.csv` - All parsed P2P operations


### 3. Generate visualizations

```bash
python nccl_visualizer.py results/analysis --output-dir results/plots
```

This will generate PNG format graphics:

- `operation_summary.png` - Operations summary by type
- `rank_summary.png` - Communications by rank
- `rank_operation_heatmap.png` - Operations heatmap
- `p2p_communication_matrix.png` - P2P communication matrices
- `p2p_send_recv_summary.png` - Send/Recv summary
- `operation_timeline.png` - Operations timeline (if there are timestamps)
- `data_size_distribution.png` - Data size distribution


## Information you can obtain

### Collective Operations

- **AllReduce**: Reduction and broadcast of data among all GPUs
- **AllGather**: Gathering data from all GPUs
- **Broadcast**: Distributing data from one GPU to all others
- **ReduceScatter**: Reduction and segmented distribution
- **Reduce**: Reduction towards a specific GPU
- **AllToAll**: Complete exchange among all GPUs


### Point-to-Point (P2P) Operations

- **Send/Recv**: Direct communications between GPU pairs


### Generated Analyses

1. **By operation type**:
    - Count of each operation type
    - Total data transferred by type
    - Average, minimum, and maximum size
2. **By rank (GPU)**:
    - Total operations performed
    - Total data sent/received
    - Distribution by operation type
3. **P2P communications**:
    - Communication matrix: who sends to whom
    - Volume and count of data
    - Most active communication pairs
4. **Communication patterns**:
    - Load balancing among GPUs
    - Identify bottlenecks
    - Asymmetries in communications

## Analysis Examples

### Identify GPUs with most communication load

```bash
# The rank_summary.csv file will show:
# - Which GPU performs the most operations
# - Which GPU transfers the most data
```


### Detect P2P communication patterns

```bash
# The p2p_comm_*_matrix.csv files will show:
# - Matrix of who communicates with whom
# - Data volume between each GPU pair
```


### Analyze dominant operation types

```bash
# The collective_ops_summary.csv file will show:
# - Which operation is used the most (e.g. AllReduce in DDP)
# - How much bandwidth each type consumes
```


## Usage Tips

### For long distributed training runs

Generate logs for only one representative epoch or iteration to avoid huge files:

```python
# In your PyTorch code
if epoch == 5:  # Only log one specific epoch
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_DEBUG_SUBSYS'] = 'COLL'
else:
    os.environ['NCCL_DEBUG'] = 'WARN'
```


### For debugging hangs or timeouts

Use additional subsystems:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL,P2P,NET <command>
```


### For bandwidth analysis

Combine with timestamps to view temporal trends:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL NCCL_DEBUG_TIMESTAMP_FORMAT="[%F_%T.%3f]" <command>
```


## Troubleshooting

### "No NCCL communications found"

- Verify that `NCCL_DEBUG=INFO` is on
- Make sure to include `NCCL_DEBUG_SUBSYS=COLL` or `COLL,P2P`
- Confirm the log contains lines with "NCCL INFO"


### Very large logs

- Capture only a few iterations
- Filter the log to include only NCCL lines:

```bash
grep "NCCL INFO" nccl_full.log > nccl_filtered.log
```


### Empty visualizations

- Check that the CSV files were generated
- Make sure matplotlib and seaborn are installed
- Some plots require specific data (e.g. timeline needs timestamps)


## Use Cases

### 1. DDP (DistributedDataParallel) debugging

Identify if there is imbalance in AllReduce of gradients.

### 2. Pipeline Parallelism Analysis

Review Send/Recv operations between pipeline stages.

### 3. Tensor Parallelism Optimization

Analyze volume of AllGather and ReduceScatter.

### 4. Model Parallelism Debugging

Check communication patterns among GPUs with different model sections.

***

**Author**: Pablo Saura Sánchez
**Date**: October 2025

***