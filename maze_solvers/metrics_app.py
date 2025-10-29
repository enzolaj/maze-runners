import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import queue
import os
import time

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    HAS_MPL = True
except Exception:
    HAS_MPL = False

from metrics_simulator import run_single, aggregate_results, DEFAULT_ALGOS, DEFAULT_GENS


class MetricsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Maze Metrics Simulator")
        self.root.geometry("1200x700")

        self._build_ui()
        self.worker_thread = None
        self.result_queue = queue.Queue()
        self.current_summary = []
        self.current_raw = []

        self.root.after(200, self._poll_worker)

    def _build_ui(self):
        top = ttk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        # Controls left
        ctrl_left = ttk.Frame(top)
        ctrl_left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        self.runs_var = tk.IntVar(value=10)
        self.width_var = tk.IntVar(value=25)
        self.height_var = tk.IntVar(value=25)
        self.loop_var = tk.IntVar(value=0)
        self.rewards_var = tk.IntVar(value=0)
        self.max_weight_var = tk.IntVar(value=1)
        self.max_ticks_var = tk.IntVar(value=20000)
        self.out_dir_var = tk.StringVar(value="metrics_output")
        self.save_charts_var = tk.BooleanVar(value=True)
        self.astar_offline_var = tk.BooleanVar(value=False)

        def add_labeled_spin(parent, label, var, f=0, t=1000):
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=15).pack(side=tk.LEFT)
            sp = ttk.Spinbox(row, from_=f, to=t, textvariable=var, width=8)
            sp.pack(side=tk.LEFT)
            return sp

        add_labeled_spin(ctrl_left, "Runs", self.runs_var, 1, 100000)
        add_labeled_spin(ctrl_left, "Width", self.width_var, 2, 200)
        add_labeled_spin(ctrl_left, "Height", self.height_var, 2, 200)
        add_labeled_spin(ctrl_left, "Loop %", self.loop_var, 0, 100)
        add_labeled_spin(ctrl_left, "Rewards", self.rewards_var, 0, 50)
        add_labeled_spin(ctrl_left, "Max Weight", self.max_weight_var, 1, 50)
        add_labeled_spin(ctrl_left, "Max Ticks", self.max_ticks_var, 100, 1000000)

        # Output directory
        out_row = ttk.Frame(ctrl_left)
        out_row.pack(fill=tk.X, pady=2)
        ttk.Label(out_row, text="Output Dir", width=15).pack(side=tk.LEFT)
        out_entry = ttk.Entry(out_row, textvariable=self.out_dir_var, width=20)
        out_entry.pack(side=tk.LEFT)
        ttk.Button(out_row, text="Browse", command=self._browse_out_dir).pack(side=tk.LEFT, padx=(6,0))

        # Controls middle: Generators and Algorithms
        ctrl_mid = ttk.Frame(top)
        ctrl_mid.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(ctrl_mid, text="Generators").grid(row=0, column=0, sticky=tk.W)
        self.gen_list = tk.Listbox(ctrl_mid, selectmode=tk.MULTIPLE, height=8, exportselection=False)
        self.gen_list.grid(row=1, column=0, sticky=tk.NSEW, padx=(0, 10))
        for g in DEFAULT_GENS:
            self.gen_list.insert(tk.END, g)
        for i in range(min(4, len(DEFAULT_GENS))):
            self.gen_list.selection_set(i)

        ttk.Label(ctrl_mid, text="Algorithms").grid(row=0, column=1, sticky=tk.W)
        self.alg_list = tk.Listbox(ctrl_mid, selectmode=tk.MULTIPLE, height=8, exportselection=False)
        self.alg_list.grid(row=1, column=1, sticky=tk.NSEW)
        for a in DEFAULT_ALGOS:
            self.alg_list.insert(tk.END, a)
        for i in range(min(4, len(DEFAULT_ALGOS))):
            self.alg_list.selection_set(i)

        # Controls right: Run, progress
        ctrl_right = ttk.Frame(top)
        ctrl_right.pack(side=tk.LEFT, fill=tk.Y)

        self.run_btn = ttk.Button(ctrl_right, text="Run Simulation", command=self._on_run)
        self.run_btn.pack(anchor=tk.N, pady=(0, 6))

        self.save_btn = ttk.Button(ctrl_right, text="Export CSVs", command=self._export_csv, state=tk.DISABLED)
        self.save_btn.pack(anchor=tk.N, pady=(0, 12))

        self.save_charts_chk = ttk.Checkbutton(ctrl_right, text="Save charts to out_dir", variable=self.save_charts_var)
        self.save_charts_chk.pack(anchor=tk.W, pady=(0,6))

        self.astar_offline_chk = ttk.Checkbutton(ctrl_right, text="A* offline (knows maze)", variable=self.astar_offline_var)
        self.astar_offline_chk.pack(anchor=tk.W, pady=(0,6))

        ttk.Label(ctrl_right, text="Status").pack(anchor=tk.W)
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(ctrl_right, textvariable=self.status_var, width=40).pack(anchor=tk.W)

        self.progress = ttk.Progressbar(ctrl_right, orient=tk.HORIZONTAL, mode='determinate', length=250)
        self.progress.pack(anchor=tk.W, pady=(6, 0))

        # Bottom: Notebook with table and charts
        bottom = ttk.Notebook(self.root)
        bottom.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Summary tab
        self.table_frame = ttk.Frame(bottom)
        bottom.add(self.table_frame, text="Summary")
        cols = ["gen_method", "algorithm", "count", "finished_rate",
                "elapsed_sec_avg", "algorithm_steps_avg", "unique_explored_avg",
                "min_path_cost_avg", "total_path_cost_avg", "total_path_length_avg",
                "nodes_expanded_avg", "frontier_max_avg"]
        self.tree = ttk.Treeview(self.table_frame, columns=cols, show='headings')
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=120, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True)

        # Charts tab
        self.charts_frame = ttk.Frame(bottom)
        bottom.add(self.charts_frame, text="Charts")

        metric_row = ttk.Frame(self.charts_frame)
        metric_row.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        ttk.Label(metric_row, text="Metric:").pack(side=tk.LEFT)
        self.metric_var = tk.StringVar(value="algorithm_steps_avg")
        self.metric_combo = ttk.Combobox(metric_row, textvariable=self.metric_var,
                                         values=[
                                             "elapsed_sec_avg", "ticks_avg", "algorithm_steps_avg", "unique_explored_avg",
                                             "min_path_cost_avg", "total_path_cost_avg", "total_path_length_avg",
                                             "nodes_expanded_avg", "frontier_max_avg"
                                         ], state="readonly", width=28)
        self.metric_combo.pack(side=tk.LEFT, padx=6)
        self.metric_combo.bind("<<ComboboxSelected>>", lambda e: self._render_chart())

        # Filters for a single combo view
        ttk.Label(metric_row, text="Generator:").pack(side=tk.LEFT, padx=(12,2))
        self.filter_gen_var = tk.StringVar(value="All")
        self.filter_gen_combo = ttk.Combobox(metric_row, textvariable=self.filter_gen_var, state="readonly", width=24)
        self.filter_gen_combo.pack(side=tk.LEFT)
        self.filter_gen_combo.bind("<<ComboboxSelected>>", lambda e: (self._update_table(), self._render_chart(), self._update_details_table(), self._render_combo_chart()))

        ttk.Label(metric_row, text="Algorithm:").pack(side=tk.LEFT, padx=(12,2))
        self.filter_algo_var = tk.StringVar(value="All")
        self.filter_algo_combo = ttk.Combobox(metric_row, textvariable=self.filter_algo_var, state="readonly", width=24)
        self.filter_algo_combo.pack(side=tk.LEFT)
        self.filter_algo_combo.bind("<<ComboboxSelected>>", lambda e: (self._update_table(), self._render_chart(), self._update_details_table(), self._render_combo_chart()))

        if not HAS_MPL:
            ttk.Label(self.charts_frame, text="matplotlib not installed - charts disabled").pack(pady=20)
            self.canvas = None
        else:
            fig = Figure(figsize=(8, 4), dpi=100)
            self.ax = fig.add_subplot(111)
            self.canvas = FigureCanvasTkAgg(fig, master=self.charts_frame)
            self.canvas_widget = self.canvas.get_tk_widget()
            self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Details tab (raw rows for selected combo)
        self.details_frame = ttk.Frame(bottom)
        bottom.add(self.details_frame, text="Combo Details")
        self.details_tree = ttk.Treeview(self.details_frame, show='headings')
        self.details_tree.pack(fill=tk.BOTH, expand=True)

        # Combo charts tab (all metrics for a single selected combo)
        self.combo_charts_frame = ttk.Frame(bottom)
        bottom.add(self.combo_charts_frame, text="Combo Charts")
        if not HAS_MPL:
            ttk.Label(self.combo_charts_frame, text="matplotlib not installed - charts disabled").pack(pady=20)
            self.combo_canvas = None
        else:
            fig2 = Figure(figsize=(8, 4), dpi=100)
            self.combo_ax = fig2.add_subplot(111)
            self.combo_canvas = FigureCanvasTkAgg(fig2, master=self.combo_charts_frame)
            self.combo_canvas_widget = self.combo_canvas.get_tk_widget()
            self.combo_canvas_widget.pack(fill=tk.BOTH, expand=True)

    def _on_run(self):
        if self.worker_thread and self.worker_thread.is_alive():
            return
        try:
            runs = int(self.runs_var.get())
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            loop_percent = int(self.loop_var.get())
            num_rewards = int(self.rewards_var.get())
            max_weight = int(self.max_weight_var.get())
            max_ticks = int(self.max_ticks_var.get())
        except Exception:
            messagebox.showerror("Invalid input", "Please enter valid numeric values.")
            return

        gens = [self.gen_list.get(i) for i in self.gen_list.curselection()]
        algos = [self.alg_list.get(i) for i in self.alg_list.curselection()]
        if not gens or not algos:
            messagebox.showwarning("Selection", "Please select at least one generator and one algorithm.")
            return

        total_jobs = len(gens) * len(algos) * runs
        self.progress['maximum'] = total_jobs
        self.progress['value'] = 0
        self.status_var.set("Running...")
        self.run_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)

        # Start worker
        args = {
            'runs': runs,
            'width': width,
            'height': height,
            'loop_percent': loop_percent,
            'num_rewards': num_rewards,
            'max_weight': max_weight,
            'max_ticks': max_ticks,
            'gens': gens,
            'algos': algos,
            'out_dir': self.out_dir_var.get(),
            'save_charts': bool(self.save_charts_var.get()),
            'astar_offline': bool(self.astar_offline_var.get()),
        }
        self.worker_thread = threading.Thread(target=self._worker_run, args=(args,), daemon=True)
        self.worker_thread.start()

    def _worker_run(self, args):
        runs = args['runs']
        width = args['width']
        height = args['height']
        loop_percent = args['loop_percent']
        num_rewards = args['num_rewards']
        max_weight = args['max_weight']
        max_ticks = args['max_ticks']
        gens = args['gens']
        algos = args['algos']
        out_dir = args['out_dir']
        save_charts = args['save_charts']
        astar_offline = args.get('astar_offline', False)
        seed_base = int(time.time())

        all_rows = []
        job_done = 0
        total_jobs = len(gens) * len(algos) * runs
        for gen in gens:
            for algo in algos:
                for i in range(runs):
                    seed = seed_base + i
                    res = run_single(width, height, gen, algo, loop_percent, num_rewards, max_weight, max_ticks=max_ticks, seed=seed, astar_offline=astar_offline)
                    all_rows.append(res)
                    job_done += 1
                    self.result_queue.put(('progress', job_done, total_jobs))

        summary = aggregate_results(all_rows)
        expanded = []
        for row in summary:
            gen, algo = row["group"]
            new_row = {k: v for k, v in row.items() if k != "group"}
            new_row["gen_method"] = gen
            new_row["algorithm"] = algo
            expanded.append(new_row)

        # Optionally save charts and CSVs
        if save_charts and HAS_MPL:
            try:
                os.makedirs(out_dir, exist_ok=True)
                from metrics_simulator import plot_metric
                for metric in [
                    "elapsed_sec_avg",
                    "algorithm_steps_avg",
                    "unique_explored_avg",
                    "min_path_cost_avg",
                    "total_path_cost_avg",
                    "total_path_length_avg",
                    "nodes_expanded_avg",
                    "frontier_max_avg",
                ]:
                    plot_metric(expanded, metric, os.path.join(out_dir, f"{metric}.png"))
            except Exception:
                pass

        self.result_queue.put(('done', all_rows, expanded))

    def _poll_worker(self):
        try:
            while True:
                msg = self.result_queue.get_nowait()
                if msg[0] == 'progress':
                    done, total = msg[1], msg[2]
                    self.progress['value'] = done
                    self.status_var.set(f"Running... {done}/{total}")
                elif msg[0] == 'done':
                    self.current_raw = msg[1]
                    self.current_summary = msg[2]
                    self._populate_filters()
                    self._setup_details_table()
                    self._update_table()
                    self._render_chart()
                    self._update_details_table()
                    self._render_combo_chart()
                    self.status_var.set("Completed")
                    self.run_btn.config(state=tk.NORMAL)
                    self.save_btn.config(state=tk.NORMAL)
        except queue.Empty:
            pass
        self.root.after(200, self._poll_worker)

    def _update_table(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        rows = self._filtered_summary()
        for row in rows:
            ordered = {c: row.get(c, '') for c in self.tree['columns']}
            vals = [ordered[c] for c in self.tree['columns']]
            self.tree.insert('', tk.END, values=vals)

    def _render_chart(self):
        if not HAS_MPL or not self.current_summary:
            return
        metric = self.metric_var.get()
        rows = self._filtered_summary()
        labels = []
        values = []
        for row in rows:
            labels.append(f"{row.get('algorithm','')}\n({row.get('gen_method','')})")
            values.append(row.get(metric, 0))
        self.ax.clear()
        bars = self.ax.bar(range(len(values)), values)
        self.ax.set_xticks(range(len(values)))
        self.ax.set_xticklabels(labels, rotation=45, ha='right')
        self.ax.set_ylabel(metric)
        self.ax.set_title("Average metric by (Algorithm, Generator)")
        self.ax.figure.tight_layout()
        # Annotate values on bars
        try:
            for bar, v in zip(bars, values):
                h = bar.get_height()
                label = self._format_value(v)
                self.ax.text(bar.get_x() + bar.get_width()/2.0, h, label, ha='center', va='bottom', fontsize=8)
        except Exception:
            pass
        self.canvas.draw_idle()

    def _render_combo_chart(self):
        if not HAS_MPL or not self.current_summary:
            return
        # Require a single combo selection
        gen = self.filter_gen_var.get()
        algo = self.filter_algo_var.get()
        if gen == "All" or algo == "All":
            if hasattr(self, 'combo_ax'):
                self.combo_ax.clear()
                self.combo_ax.text(0.5, 0.5, "Select a specific Generator AND Algorithm", ha='center', va='center')
                self.combo_ax.set_axis_off()
                self.combo_canvas.draw_idle()
            return
        # Find matching summary row
        row = None
        for r in self.current_summary:
            if r.get('gen_method','') == gen and r.get('algorithm','') == algo:
                row = r; break
        if row is None or not hasattr(self, 'combo_ax'):
            return
        # Metrics to show
        metrics = [
            "elapsed_sec_avg", "ticks_avg", "algorithm_steps_avg", "unique_explored_avg",
            "min_path_cost_avg", "total_path_cost_avg", "total_path_length_avg",
            "nodes_expanded_avg", "frontier_max_avg"
        ]
        labels = metrics
        values = [row.get(m, 0) for m in metrics]
        self.combo_ax.clear()
        bars = self.combo_ax.bar(range(len(values)), values)
        self.combo_ax.set_xticks(range(len(values)))
        self.combo_ax.set_xticklabels(labels, rotation=45, ha='right')
        self.combo_ax.set_ylabel('value')
        self.combo_ax.set_title(f"All metrics for {algo} on {gen}")
        self.combo_ax.figure.tight_layout()
        # Annotate values on bars
        try:
            for bar, v in zip(bars, values):
                h = bar.get_height()
                label = self._format_value(v)
                self.combo_ax.text(bar.get_x() + bar.get_width()/2.0, h, label, ha='center', va='bottom', fontsize=8)
        except Exception:
            pass
        self.combo_canvas.draw_idle()

    def _format_value(self, v):
        try:
            iv = int(round(float(v)))
            if abs(float(v) - iv) < 1e-9:
                return str(iv)
        except Exception:
            pass
        try:
            return f"{float(v):.2f}"
        except Exception:
            return str(v)

    def _populate_filters(self):
        if not self.current_summary:
            self.filter_gen_combo.configure(values=["All"]) ; self.filter_gen_var.set("All")
            self.filter_algo_combo.configure(values=["All"]) ; self.filter_algo_var.set("All")
            return
        gens = sorted({row.get('gen_method','') for row in self.current_summary if 'gen_method' in row})
        algos = sorted({row.get('algorithm','') for row in self.current_summary if 'algorithm' in row})
        self.filter_gen_combo.configure(values=["All"] + gens)
        self.filter_algo_combo.configure(values=["All"] + algos)
        if self.filter_gen_var.get() not in (["All"] + gens): self.filter_gen_var.set("All")
        if self.filter_algo_var.get() not in (["All"] + algos): self.filter_algo_var.set("All")

    def _filtered_summary(self):
        rows = self.current_summary
        gen = getattr(self, 'filter_gen_var', tk.StringVar(value="All")).get()
        algo = getattr(self, 'filter_algo_var', tk.StringVar(value="All")).get()
        if gen and gen != "All":
            rows = [r for r in rows if r.get('gen_method','') == gen]
        if algo and algo != "All":
            rows = [r for r in rows if r.get('algorithm','') == algo]
        return rows

    def _setup_details_table(self):
        for i in self.details_tree.get_children():
            self.details_tree.delete(i)
        if not self.current_raw:
            return
        cols = list(self.current_raw[0].keys())
        self.details_tree.configure(columns=cols)
        for c in cols:
            self.details_tree.heading(c, text=c)
            self.details_tree.column(c, width=110, anchor=tk.CENTER)

    def _update_details_table(self):
        for i in self.details_tree.get_children():
            self.details_tree.delete(i)
        rows = self.current_raw
        gen = getattr(self, 'filter_gen_var', tk.StringVar(value="All")).get()
        algo = getattr(self, 'filter_algo_var', tk.StringVar(value="All")).get()
        if gen and gen != "All":
            rows = [r for r in rows if r.get('gen_method','') == gen]
        if algo and algo != "All":
            rows = [r for r in rows if r.get('algorithm','') == algo]
        if not rows:
            return
        # Ensure columns match
        cols = list(self.details_tree['columns'])
        if not cols:
            cols = list(rows[0].keys())
            self.details_tree.configure(columns=cols)
            for c in cols:
                self.details_tree.heading(c, text=c)
                self.details_tree.column(c, width=110, anchor=tk.CENTER)
        for r in rows:
            vals = [r.get(c, '') for c in self.details_tree['columns']]
            self.details_tree.insert('', tk.END, values=vals)

    def _export_csv(self):
        if not self.current_raw or not self.current_summary:
            return
        out_dir = filedialog.askdirectory(title="Select output directory")
        if not out_dir:
            return
        # raw
        raw_path = os.path.join(out_dir, "raw_results.csv")
        import csv
        with open(raw_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.current_raw[0].keys()))
            writer.writeheader()
            for r in self.current_raw:
                writer.writerow(r)
        # summary
        sum_path = os.path.join(out_dir, "summary.csv")
        with open(sum_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.current_summary[0].keys()))
            writer.writeheader()
            for r in self.current_summary:
                writer.writerow(r)
        messagebox.showinfo("Export", f"CSV files saved to {out_dir}")

    def _browse_out_dir(self):
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            self.out_dir_var.set(d)


if __name__ == "__main__":
    root = tk.Tk()
    app = MetricsApp(root)
    root.mainloop()


