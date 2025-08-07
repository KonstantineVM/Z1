"""
Results management for Kalman filter analysis.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


@dataclass
class ResultsMetadata:
    """Metadata for analysis results."""
    timestamp: str
    config_path: str
    target_series: str
    data_range: tuple
    n_observations: int
    model_type: str = "HierarchicalKalmanFilter"
    notes: Dict[str, Any] = field(default_factory=dict)


class ResultsManager:
    """Manages all outputs from the Kalman filter analysis pipeline."""
    
    def __init__(self, output_dir: str = "output", 
                 timestamp_outputs: bool = True):
        """
        Initialize results manager.
        
        Parameters
        ----------
        output_dir : str
            Base directory for outputs
        timestamp_outputs : bool
            Whether to add timestamps to output filenames
        """
        self.output_dir = Path(output_dir)
        self.timestamp_outputs = timestamp_outputs
        self.logger = logging.getLogger(__name__)
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output subdirectories
        self._create_output_structure()
        
        # Storage for results
        self.results = {}
        self.dataframes = {}
        self.plots = {}
        self.metadata = None
        
    def _create_output_structure(self):
        """Create directory structure for outputs."""
        # Base directories
        self.dirs = {
            'root': self.output_dir,
            'data': self.output_dir / 'data',
            'plots': self.output_dir / 'plots',
            'models': self.output_dir / 'models',
            'diagnostics': self.output_dir / 'diagnostics',
            'reports': self.output_dir / 'reports'
        }
        
        # Add timestamp subdirectories if enabled
        if self.timestamp_outputs:
            run_dir = self.output_dir / f'run_{self.timestamp}'
            self.dirs = {
                'root': run_dir,
                'data': run_dir / 'data',
                'plots': run_dir / 'plots',
                'models': run_dir / 'models',
                'diagnostics': run_dir / 'diagnostics',
                'reports': run_dir / 'reports'
            }
        
        # Create all directories
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Created output directory structure at: {self.dirs['root']}")
    
    def set_metadata(self, config_path: str, target_series: str,
                    data: pd.DataFrame, **kwargs):
        """Set metadata for this analysis run."""
        self.metadata = ResultsMetadata(
            timestamp=self.timestamp,
            config_path=config_path,
            target_series=target_series,
            data_range=(data.index[0].strftime("%Y-%m-%d"),
                       data.index[-1].strftime("%Y-%m-%d")),
            n_observations=len(data),
            notes=kwargs
        )
    
    def add_result(self, name: str, result: Any, category: str = 'general'):
        """Add a result to storage."""
        if category not in self.results:
            self.results[category] = {}
        self.results[category][name] = result
        self.logger.debug(f"Added result '{name}' to category '{category}'")
    
    def add_dataframe(self, name: str, df: pd.DataFrame, 
                     save_csv: bool = True, subdir: Optional[str] = None):
        """
        Add a DataFrame result.
        
        Parameters
        ----------
        name : str
            Name for the dataframe
        df : pd.DataFrame
            The dataframe to store
        save_csv : bool
            Whether to immediately save as CSV
        subdir : str, optional
            Subdirectory within data folder
        """
        self.dataframes[name] = df
        
        if save_csv:
            save_dir = self.dirs['data']
            if subdir:
                save_dir = save_dir / subdir
                save_dir.mkdir(exist_ok=True)
            
            filename = f"{name}_{self.timestamp}.csv" if self.timestamp_outputs else f"{name}.csv"
            filepath = save_dir / filename
            
            df.to_csv(filepath)
            self.logger.info(f"Saved DataFrame '{name}' to: {filepath}")
    
    def add_plot(self, name: str, fig: plt.Figure, 
                save_plot: bool = True, subdir: Optional[str] = None,
                dpi: int = 150):
        """
        Add a plot result.
        
        Parameters
        ----------
        name : str
            Name for the plot
        fig : plt.Figure
            The matplotlib figure
        save_plot : bool
            Whether to immediately save
        subdir : str, optional
            Subdirectory within plots folder
        dpi : int
            DPI for saving
        """
        self.plots[name] = fig
        
        if save_plot:
            save_dir = self.dirs['plots']
            if subdir:
                save_dir = save_dir / subdir
                save_dir.mkdir(exist_ok=True)
            
            filename = f"{name}_{self.timestamp}.png" if self.timestamp_outputs else f"{name}.png"
            filepath = save_dir / filename
            
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            self.logger.info(f"Saved plot '{name}' to: {filepath}")
    
    def save_model_results(self, model_name: str, results_dict: Dict[str, Any]):
        """Save model results including parameters and diagnostics."""
        filename = f"{model_name}_{self.timestamp}.json" if self.timestamp_outputs else f"{model_name}.json"
        filepath = self.dirs['models'] / filename
        
        # Convert numpy arrays and pandas objects for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj
        
        # Recursively convert the dictionary
        json_safe_dict = json.loads(
            json.dumps(results_dict, default=convert_for_json)
        )
        
        with open(filepath, 'w') as f:
            json.dump(json_safe_dict, f, indent=2)
        
        self.logger.info(f"Saved model results to: {filepath}")
    
    def save_diagnostics(self, diagnostics: Dict[str, Any]):
        """Save diagnostic information."""
        filename = f"diagnostics_{self.timestamp}.json" if self.timestamp_outputs else "diagnostics.json"
        filepath = self.dirs['diagnostics'] / filename
        
        # Add metadata
        diagnostics['metadata'] = {
            'timestamp': self.metadata.timestamp if self.metadata else self.timestamp,
            'target_series': self.metadata.target_series if self.metadata else 'unknown',
            'n_observations': self.metadata.n_observations if self.metadata else 0
        }
        
        with open(filepath, 'w') as f:
            json.dump(diagnostics, f, indent=2, default=str)
        
        self.logger.info(f"Saved diagnostics to: {filepath}")
    
    def create_summary_report(self):
        """Create a summary report of all results."""
        report_lines = [
            "# Kalman Filter Analysis Summary Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Output Directory: {self.dirs['root']}\n"
        ]
        
        # Add metadata
        if self.metadata:
            report_lines.extend([
                "## Analysis Configuration",
                f"- Target Series: {self.metadata.target_series}",
                f"- Data Range: {self.metadata.data_range[0]} to {self.metadata.data_range[1]}",
                f"- Observations: {self.metadata.n_observations}",
                f"- Model Type: {self.metadata.model_type}\n"
            ])
        
        # Add results summary
        report_lines.append("## Results Summary")
        
        # DataFrames
        if self.dataframes:
            report_lines.append("\n### DataFrames Saved:")
            for name, df in self.dataframes.items():
                report_lines.append(f"- {name}: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Plots
        if self.plots:
            report_lines.append("\n### Plots Generated:")
            for name in self.plots.keys():
                report_lines.append(f"- {name}")
        
        # General results
        if self.results:
            report_lines.append("\n### Analysis Results:")
            for category, items in self.results.items():
                report_lines.append(f"\n#### {category.capitalize()}:")
                for name, value in items.items():
                    if isinstance(value, (int, float)):
                        report_lines.append(f"- {name}: {value:.4f}")
                    elif isinstance(value, dict):
                        report_lines.append(f"- {name}: {len(value)} items")
                    else:
                        report_lines.append(f"- {name}: {type(value).__name__}")
        
        # Save report
        report_filename = f"summary_report_{self.timestamp}.md" if self.timestamp_outputs else "summary_report.md"
        report_path = self.dirs['reports'] / report_filename
        
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"Created summary report: {report_path}")
        
        return report_path
    
    def export_key_results(self, results_dict: Dict[str, Any]):
        """Export key results to a single JSON file for easy access."""
        filename = f"key_results_{self.timestamp}.json" if self.timestamp_outputs else "key_results.json"
        filepath = self.dirs['root'] / filename
        
        # Convert for JSON
        def convert_for_json(obj):
            if isinstance(obj, (np.ndarray, pd.Series, pd.DataFrame)):
                return None  # Skip complex objects
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif hasattr(obj, 'isoformat'):
                return obj.isoformat()
            return obj
        
        clean_dict = json.loads(
            json.dumps(results_dict, default=convert_for_json)
        )
        
        with open(filepath, 'w') as f:
            json.dump(clean_dict, f, indent=2)
        
        self.logger.info(f"Exported key results to: {filepath}")
    
    def get_output_path(self, filename: str, subdir: str = 'data') -> Path:
        """Get the full path for an output file."""
        if subdir not in self.dirs:
            raise ValueError(f"Unknown subdirectory: {subdir}")
        
        if self.timestamp_outputs and not filename.endswith(self.timestamp):
            base, ext = filename.rsplit('.', 1)
            filename = f"{base}_{self.timestamp}.{ext}"
        
        return self.dirs[subdir] / filename
    
    def list_outputs(self) -> Dict[str, List[str]]:
        """List all outputs created."""
        outputs = {}
        
        for name, dir_path in self.dirs.items():
            if name == 'root':
                continue
            
            files = list(dir_path.rglob('*'))
            files = [f for f in files if f.is_file()]
            outputs[name] = [str(f.relative_to(self.output_dir)) for f in files]
        
        return outputs
    
    def __repr__(self):
        return f"ResultsManager(output_dir='{self.output_dir}', timestamp='{self.timestamp}')"
