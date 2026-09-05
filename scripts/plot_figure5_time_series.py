"""Compatibility entry point for the renumbered strong-disturbance time-series figure."""
# .venv/bin/python scripts/plot_figure5_time_series.py \
#   --model train_data1/train_lambda0.0_0.1_0.3_0.5_0.8_1.0_20260518_175113/train_lambda0.1_20260518_185047 \
#   --out_dir data/figure5 \
#   --start_idx 29000 \
#   --output_prefix figure5-2_time_series

from plot_figure6_time_series import main


if __name__ == "__main__":
    main()
