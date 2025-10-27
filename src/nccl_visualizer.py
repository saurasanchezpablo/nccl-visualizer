#!/usr/bin/env python3
"""
NCCL Analysis Visualizer
Genera visualizaciones a partir de los CSVs generados por nccl_parser.py

Uso:
    python nccl_visualizer.py <prefix> [--output-dir <dir>]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path

class NCCLVisualizer:
    """Genera visualizaciones de análisis NCCL"""

    def __init__(self, prefix, output_dir='plots'):
        self.prefix = prefix
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Carga los archivos CSV generados por el parser"""
        self.data = {}

        files = {
            'collective_ops': f'{self.prefix}_collective_ops_summary.csv',
            'collective_rank': f'{self.prefix}_collective_rank_summary.csv',
            'rank_op_matrix': f'{self.prefix}_collective_rank_op_matrix.csv',
            'p2p_count': f'{self.prefix}_p2p_comm_count_matrix.csv',
            'p2p_data': f'{self.prefix}_p2p_comm_data_matrix.csv',
            'p2p_summary': f'{self.prefix}_p2p_rank_summary.csv',
            'all_collective': f'{self.prefix}_all_collective_ops.csv',
            'all_p2p': f'{self.prefix}_all_p2p_ops.csv',
        }

        for key, filename in files.items():
            if os.path.exists(filename):
                self.data[key] = pd.read_csv(filename, index_col=0 if key != 'all_collective' and key != 'all_p2p' else None)
                print(f"✓ Cargado: {filename}")
            else:
                print(f"⚠ No encontrado: {filename}")

    def plot_operation_summary(self):
        """Gráfico de resumen de operaciones colectivas"""
        if 'collective_ops' not in self.data:
            return

        df = self.data['collective_ops']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Gráfico de barras: cantidad de operaciones por tipo
        ax1 = axes[0]
        df['count'].plot(kind='bar', ax=ax1, color='steelblue')
        ax1.set_title('Cantidad de Operaciones por Tipo', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Tipo de Operación')
        ax1.set_ylabel('Cantidad')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)

        # Gráfico de barras: datos transferidos por tipo
        ax2 = axes[1]
        df['total_mb'].plot(kind='bar', ax=ax2, color='coral')
        ax2.set_title('Datos Transferidos por Tipo de Operación', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Tipo de Operación')
        ax2.set_ylabel('Datos (MB)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_file = f'{self.output_dir}/operation_summary.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Guardado: {output_file}")
        plt.close()

    def plot_rank_communication(self):
        """Gráfico de comunicaciones por rank"""
        if 'collective_rank' not in self.data:
            return

        df = self.data['collective_rank']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Operaciones por rank
        ax1 = axes[0]
        df['total_operations'].plot(kind='bar', ax=ax1, color='mediumseagreen')
        ax1.set_title('Operaciones Colectivas por Rank', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Rank')
        ax1.set_ylabel('Número de Operaciones')
        ax1.tick_params(axis='x', rotation=0)
        ax1.grid(axis='y', alpha=0.3)

        # Datos por rank
        ax2 = axes[1]
        df['total_data_mb'].plot(kind='bar', ax=ax2, color='darkorange')
        ax2.set_title('Datos Transferidos por Rank', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Rank')
        ax2.set_ylabel('Datos (MB)')
        ax2.tick_params(axis='x', rotation=0)
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_file = f'{self.output_dir}/rank_summary.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Guardado: {output_file}")
        plt.close()

    def plot_rank_operation_matrix(self):
        """Heatmap de operaciones por rank y tipo"""
        if 'rank_op_matrix' not in self.data:
            return

        df = self.data['rank_op_matrix']

        plt.figure(figsize=(10, 6))
        sns.heatmap(df, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Cantidad'})
        plt.title('Matriz de Operaciones: Rank vs Tipo de Operación', fontsize=12, fontweight='bold')
        plt.xlabel('Tipo de Operación')
        plt.ylabel('Rank')
        plt.tight_layout()

        output_file = f'{self.output_dir}/rank_operation_heatmap.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Guardado: {output_file}")
        plt.close()

    def plot_p2p_communication_matrix(self):
        """Heatmaps de comunicación P2P"""
        if 'p2p_count' not in self.data:
            return

        count_df = self.data['p2p_count']
        data_df = self.data.get('p2p_data')

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Matriz de cantidad de comunicaciones
        ax1 = axes[0]
        sns.heatmap(count_df, annot=True, fmt='g', cmap='Blues', ax=ax1, 
                   cbar_kws={'label': 'Número de comunicaciones'})
        ax1.set_title('Matriz de Comunicaciones P2P (Cantidad)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Rank Destino')
        ax1.set_ylabel('Rank Origen')

        # Matriz de datos transferidos
        if data_df is not None:
            ax2 = axes[1]
            sns.heatmap(data_df, annot=True, fmt='.1f', cmap='Reds', ax=ax2,
                       cbar_kws={'label': 'Datos (MB)'})
            ax2.set_title('Matriz de Comunicaciones P2P (Datos MB)', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Rank Destino')
            ax2.set_ylabel('Rank Origen')

        plt.tight_layout()
        output_file = f'{self.output_dir}/p2p_communication_matrix.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Guardado: {output_file}")
        plt.close()

    def plot_p2p_summary(self):
        """Gráfico resumen de Send/Recv por rank"""
        if 'p2p_summary' not in self.data:
            return

        df = self.data['p2p_summary']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Cantidad de Send/Recv
        ax1 = axes[0]
        df[['send_count', 'recv_count']].plot(kind='bar', ax=ax1, color=['steelblue', 'coral'])
        ax1.set_title('Operaciones Send/Recv por Rank', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Rank')
        ax1.set_ylabel('Cantidad de Operaciones')
        ax1.legend(['Send', 'Recv'])
        ax1.tick_params(axis='x', rotation=0)
        ax1.grid(axis='y', alpha=0.3)

        # Datos Send/Recv
        ax2 = axes[1]
        df[['send_data_mb', 'recv_data_mb']].plot(kind='bar', ax=ax2, color=['mediumseagreen', 'darkorange'])
        ax2.set_title('Datos Send/Recv por Rank', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Rank')
        ax2.set_ylabel('Datos (MB)')
        ax2.legend(['Send (MB)', 'Recv (MB)'])
        ax2.tick_params(axis='x', rotation=0)
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_file = f'{self.output_dir}/p2p_send_recv_summary.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Guardado: {output_file}")
        plt.close()

    def plot_operation_timeline(self):
        """Timeline de operaciones si hay timestamps"""
        if 'all_collective' not in self.data:
            return

        df = self.data['all_collective']

        # Verificar si hay timestamps válidos
        if 'timestamp' not in df.columns or df['timestamp'].isna().all():
            print("⚠ No hay timestamps disponibles para timeline")
            return

        # Filtrar filas con timestamp válido
        df_time = df[df['timestamp'].notna()].copy()
        if len(df_time) == 0:
            return

        # Convertir timestamp a datetime
        df_time['timestamp'] = pd.to_datetime(df_time['timestamp'])
        df_time = df_time.sort_values('timestamp')
        df_time['time_seconds'] = (df_time['timestamp'] - df_time['timestamp'].min()).dt.total_seconds()

        plt.figure(figsize=(14, 6))

        # Scatter plot por operación
        for op in df_time['operation'].unique():
            df_op = df_time[df_time['operation'] == op]
            plt.scatter(df_op['time_seconds'], df_op['rank'], label=op, alpha=0.6, s=50)

        plt.xlabel('Tiempo (segundos)')
        plt.ylabel('Rank')
        plt.title('Timeline de Operaciones Colectivas', fontsize=12, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_file = f'{self.output_dir}/operation_timeline.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Guardado: {output_file}")
        plt.close()

    def plot_data_size_distribution(self):
        """Distribución de tamaños de datos transferidos"""
        if 'all_collective' not in self.data:
            return

        df = self.data['all_collective']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histograma de tamaños
        ax1 = axes[0]
        df['data_size_mb'].hist(bins=50, ax=ax1, color='steelblue', edgecolor='black')
        ax1.set_title('Distribución de Tamaños de Datos', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Tamaño de Datos (MB)')
        ax1.set_ylabel('Frecuencia')
        ax1.grid(axis='y', alpha=0.3)

        # Box plot por tipo de operación
        ax2 = axes[1]
        df.boxplot(column='data_size_mb', by='operation', ax=ax2)
        ax2.set_title('Distribución de Tamaños por Tipo de Operación', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Tipo de Operación')
        ax2.set_ylabel('Tamaño de Datos (MB)')
        plt.suptitle('')  # Remover título automático
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        output_file = f'{self.output_dir}/data_size_distribution.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Guardado: {output_file}")
        plt.close()

    def generate_all_plots(self):
        """Genera todas las visualizaciones"""
        print(f"\nGenerando visualizaciones en: {self.output_dir}/")
        print("-" * 60)

        self.plot_operation_summary()
        self.plot_rank_communication()
        self.plot_rank_operation_matrix()
        self.plot_p2p_communication_matrix()
        self.plot_p2p_summary()
        self.plot_operation_timeline()
        self.plot_data_size_distribution()

        print("-" * 60)
        print("\n✓ Todas las visualizaciones generadas")


def main():
    parser = argparse.ArgumentParser(
        description='Genera visualizaciones de análisis NCCL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python nccl_visualizer.py nccl_analysis
  python nccl_visualizer.py nccl_analysis --output-dir my_plots
        """
    )

    parser.add_argument('prefix', help='Prefijo de archivos CSV generados por nccl_parser.py')
    parser.add_argument('--output-dir', '-o', default='plots',
                       help='Directorio para guardar las visualizaciones (default: plots)')

    args = parser.parse_args()

    visualizer = NCCLVisualizer(args.prefix, args.output_dir)
    visualizer.load_data()
    visualizer.generate_all_plots()


if __name__ == '__main__':
    main()
