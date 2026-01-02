"""
Visualization Module - Generate EDA plots
FIXED: Typos, better error handling, improved plot quality
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict
import mlflow

from utils import ensure_directory
from utils.logging_mixin import LoggerMixin  


class Visualizations(LoggerMixin):  
    """Generate comprehensive EDA visualizations."""
    
    def __init__(self, config: Dict):
        """
        Initialize Visualizations.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = self.setup_class_logger('visualizations', config, 'logging')  
        
        # Plot settings
        self.figsize_univariate = tuple(self.config['visualization']['figure_size_univariate'])
        self.figsize_multivariate = tuple(self.config['visualization']['figure_size_multivariate'])
        self.save_plots = self.config['visualization']['save_plots']
        self.output_dir = Path(self.config['output']['plots_dir'])
        
        # Ensure output directory exists
        ensure_directory(self.output_dir)
        
        # Setup plot style
        self.setup_plot_style()
    
    def setup_plot_style(self) -> None:
        """Configure matplotlib and seaborn plot styling."""
        style = self.config['visualization'].get('style', 'seaborn-v0_8-darkgrid')
        context = self.config['visualization'].get('context', 'notebook')
        
        try:
            plt.style.use(style)
            self.logger.debug(f'Using plot style: {style}')
        except OSError:
            plt.style.use('default')
            self.logger.warning(f'Style {style} not found, using default')
        
        sns.set_context(context)
    
    def plot_numeric_distributions(self, df: pd.DataFrame) -> None:
        """
        Plot distributions for numeric columns.
        
        Args:
            df: Input DataFrame
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            self.logger.warning('No numeric columns to plot')
            return
        
        n_cols = len(numeric_cols)
        n_rows = (n_cols + 2) // 3  # Ceiling division
        
        self.logger.info(f'Plotting distributions for {n_cols} numeric columns...')
        
        fig, axes = plt.subplots(nrows=n_rows, ncols=3, figsize=self.figsize_multivariate)
        axes = axes.flatten() if n_cols > 1 else [axes]
        
        kde = self.config['visualization']['histogram'].get('kde', True)
        alpha = self.config['visualization']['histogram'].get('alpha', 0.7)
        
        for idx, col in enumerate(numeric_cols):
            try:
                sns.histplot(
                    data=df,
                    x=col,
                    kde=kde,
                    ax=axes[idx],
                    color='purple',
                    alpha=alpha
                )
                axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(True, alpha=0.3)
            except Exception as e:
                self.logger.error(f'Error plotting {col}: {e}')
                axes[idx].text(0.5, 0.5, f'Error: {str(e)[:50]}', 
                            ha='center', va='center', transform=axes[idx].transAxes)
        
        # Hide empty subplots
        for idx in range(n_cols, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if self.save_plots:
            output_file = self.output_dir / 'numeric_distributions.png'
            dpi = self.config['output'].get('plot_dpi', 300)
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
            mlflow.log_artifact(output_file)
            plt.close(fig)
            self.logger.info(f'✓ Saved: {output_file}')
        else:
            plt.show()
    
    def plot_categorical_distributions(self, df: pd.DataFrame) -> None:
        """
        Plot count plots for categorical columns.
        
        Args:
            df: Input DataFrame
        """
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if not categorical_cols:
            self.logger.warning('No categorical columns to plot')
            return
        
        n_cols = len(categorical_cols)
        n_rows = (n_cols + 2) // 3
        
        self.logger.info(f'Plotting frequencies for {n_cols} categorical columns...')
        
        fig, axes = plt.subplots(n_rows, 3, figsize=self.figsize_multivariate)
        axes = axes.flatten() if n_cols > 1 else [axes]
        
        for idx, col in enumerate(categorical_cols):
            try:
                ax = sns.countplot(data=df, x=col, ax=axes[idx], color='green')
                
                # Add value labels
                for container in ax.containers:
                    ax.bar_label(container, label_type='edge')
                
                ax.set_title(f'{col}', fontweight='bold')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3, axis='y')
                ax.tick_params(axis='x', rotation=45)
            except Exception as e:
                self.logger.error(f'Error plotting {col}: {e}')
        
        # Hide empty subplots
        for idx in range(n_cols, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if self.save_plots:
            output_file = self.output_dir / 'categorical_distributions.png'
            dpi = self.config['output'].get('plot_dpi', 300)
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
            mlflow.log_artifact(output_file)
            plt.close(fig)
            self.logger.info(f'✓ Saved: {output_file}')
        else:
            plt.show()
    
    def plot_boxplots(self, df: pd.DataFrame) -> None:
        """
        Plot boxplots for outlier detection.
        
        Args:
            df: Input DataFrame
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            self.logger.warning('No numeric columns for boxplots')
            return
        
        n_cols = len(numeric_cols)
        n_rows = (n_cols + 2) // 3
        
        self.logger.info(f'Plotting boxplots for {n_cols} numerical columns...')
        
        fig, axes = plt.subplots(n_rows, 3, figsize=self.figsize_multivariate)
        axes = axes.flatten() if n_cols > 1 else [axes]
        
        for idx, col in enumerate(numeric_cols):
            try:
                sns.boxplot(
                    data=df,
                    y=col,
                    ax=axes[idx],
                    color='gold',
                    linewidth=2
                )
                axes[idx].set_title(f'Boxplot - {col}', fontweight='bold')
                axes[idx].grid(True, alpha=0.3)
            except Exception as e:
                self.logger.error(f'Error plotting {col}: {e}')
        
        # Hide empty subplots
        for idx in range(n_cols, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if self.save_plots:
            output_file = self.output_dir / 'boxplots_outliers.png'
            dpi = self.config['output'].get('plot_dpi', 300)
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
            mlflow.log_artifact(output_file)
            plt.close(fig)
            self.logger.info(f'✓ Saved: {output_file}')
        else:
            plt.show()
    
    def correlation_heatmap(self, df: pd.DataFrame) -> None:
        """
        Plot correlation heatmap.
        
        Args:
            df: Input DataFrame
        """
        method = self.config['visualization']['heatmap'].get('method', 'spearman')
        self.logger.info(f'Plotting correlation heatmap ({method} method)...')
    
        try:
            corr = df.corr(numeric_only=True, method=method)
            
            # FIXED: subplot -> subplots, figize -> figsize
            fig, ax = plt.subplots(figsize=self.figsize_univariate)
            
            sns.heatmap(
                data=corr,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                ax=ax,
                cbar_kws={'label': 'Correlation'}
            )
            ax.set_title(f'Correlation Heatmap ({method.capitalize()})', fontweight='bold')
            
            plt.tight_layout()
            
            if self.save_plots:
                output_file = self.output_dir / f'correlation_heatmap_{method}.png'
                dpi = self.config['output'].get('plot_dpi', 300)
                plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
                mlflow.log_artifact(output_file)
                plt.close(fig)
                self.logger.info(f'✓ Saved: {output_file}')  
            else:
                plt.show()
        
        except Exception as e:
            self.logger.error(f'Error creating correlation heatmap: {e}', exc_info=True)


    def run_visualizations(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Run all visualizations.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with status of each visualization
        """
        results = {}
        
        self.logger.info('='*60)
        self.logger.info('RUNNING VISUALIZATIONS')
        self.logger.info('='*60)
        
        try:
            self.plot_numeric_distributions(df)
            results['numeric_distributions'] = 'success'
            
        except Exception as e:
            self.logger.error(f'Numeric distributions failed: {e}')
            results['numeric_distributions'] = f'failed: {e}'
        
        try:
            self.plot_categorical_distributions(df)
            results['categorical_distributions'] = 'success'
        except Exception as e:
            self.logger.error(f'Categorical distributions failed: {e}')
            results['categorical_distributions'] = f'failed: {e}'
        
        try:
            self.correlation_heatmap(df)
            results['heatmap'] = 'success'
        except Exception as e:
            self.logger.error(f'Heatmap failed: {e}')
            results['heatmap'] = f'failed: {e}'
        
        try:
            self.plot_boxplots(df)
            results['boxplots'] = 'success'
        except Exception as e:
            self.logger.error(f'Boxplots failed: {e}')
            results['boxplots'] = f'failed: {e}'

        
        self.logger.info('='*60)
        self.logger.info('✓ VISUALIZATIONS COMPLETED')
        self.logger.info('='*60)
        
        mlflow.log_params(results)
        return results