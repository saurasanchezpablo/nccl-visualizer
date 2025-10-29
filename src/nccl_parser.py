#!/usr/bin/env python3
"""
NCCL Debug Log Parser and Analyzer - Multi-File con Global Rank
Parsea múltiples logs de NCCL y calcula el rank global basado en:
  - Hostname del nodo
  - Rank local (0-3)

Ejemplo:
  - as07r3b19 + local_rank 0 → global_rank 0
  - as07r3b19 + local_rank 1 → global_rank 1
  - as07r3b21 + local_rank 0 → global_rank 4
  - as07r3b21 + local_rank 1 → global_rank 5

Uso:
    python nccl_parser.py ../tests/nccl_debug_* -o ../tests/results
"""

import re
import sys
import glob
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime
import argparse
import json
import os

class NCCLLogParser:
    """Parser para logs de NCCL DEBUG - Multi-file con global rank"""

    # Patrón para operaciones colectivas (timestamp opcional)
    COLLECTIVE_PATTERN = re.compile(
        r'^(?:\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\])?'  # timestamp opcional
        r'(\S+):(\d+):(\d+)\s+'  # hostname:pid:tid
        r'\[(\d+)\]\s+'  # rank local
        r'NCCL INFO\s+'
        r'(AllReduce|AllGather|Broadcast|ReduceScatter|Reduce|AllToAll):\s*'  # operation
        r'opCount\s+(\d+|[0-9a-fA-F]+)\s+'  # opCount
        r'sendbuff\s+(0x[0-9a-fA-F]+|\(nil\))\s+'  # sendbuff
        r'recvbuff\s+(0x[0-9a-fA-F]+|\(nil\))\s+'  # recvbuff
        r'count\s+(\d+)\s+'  # count
        r'datatype\s+(\d+)'  # datatype
        r'.*\[nranks=(\d+)\]'  # nranks
    )

    # Patrón para operaciones P2P Send (timestamp opcional)
    P2P_SEND_PATTERN = re.compile(
        r'^(?:\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\])?'  # timestamp opcional
        r'(\S+):(\d+):(\d+)\s+'  # hostname:pid:tid
        r'\[(\d+)\]\s+'  # rank local
        r'NCCL INFO\s+'
        r'Send:\s*'  # operation
        r'opCount\s+(\d+|[0-9a-fA-F]+)\s+'  # opCount
        r'sendbuff\s+(0x[0-9a-fA-F]+|\(nil\))\s+'  # sendbuff
        r'recvbuff\s+(0x[0-9a-fA-F]+|\(nil\))\s+'  # recvbuff
        r'count\s+(\d+)\s+'  # count
        r'datatype\s+(\d+)\s+'  # datatype
        r'.*root\s+(\d+)'  # peer rank local
        r'.*\[nranks=(\d+)\]'  # nranks
    )

    # Patrón para operaciones P2P Recv (timestamp opcional)
    P2P_RECV_PATTERN = re.compile(
        r'^(?:\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\])?'  # timestamp opcional
        r'(\S+):(\d+):(\d+)\s+'  # hostname:pid:tid
        r'\[(\d+)\]\s+'  # rank local
        r'NCCL INFO\s+'
        r'Recv:\s*'  # operation
        r'opCount\s+(\d+|[0-9a-fA-F]+)\s+'  # opCount
        r'sendbuff\s+(0x[0-9a-fA-F]+|\(nil\))\s+'  # sendbuff
        r'recvbuff\s+(0x[0-9a-fA-F]+|\(nil\))\s+'  # recvbuff
        r'count\s+(\d+)\s+'  # count
        r'datatype\s+(\d+)\s+'  # datatype
        r'.*root\s+(\d+)'  # peer rank local
        r'.*\[nranks=(\d+)\]'  # nranks
    )

    # Mapeo de tipos de datos NCCL a tamaño en bytes
    DATATYPE_SIZES = {
        0: 1,   # int8
        1: 1,   # uint8
        2: 4,   # float32
        3: 8,   # float64
        4: 2,   # float16
        5: 4,   # int32
        6: 4,   # uint32
        7: 8,   # int64
        8: 8,   # uint64
        9: 2,   # bfloat16
    }

    def __init__(self, gpus_per_node=4):
        self.communications = []
        self.p2p_communications = []
        self.parse_errors = []
        self.has_timestamps = False
        self.files_processed = []
        self.gpus_per_node = gpus_per_node

        # Mapeo de hostname a node_id
        self.hostname_to_node_id = {}
        self.next_node_id = 0

    def _get_node_id(self, hostname):
        """Obtiene un node_id único para cada hostname"""
        if hostname not in self.hostname_to_node_id:
            self.hostname_to_node_id[hostname] = self.next_node_id
            self.next_node_id += 1
        return self.hostname_to_node_id[hostname]

    def _compute_global_rank(self, hostname, local_rank):
        """Calcula el rank global: node_id * gpus_per_node + local_rank"""
        node_id = self._get_node_id(hostname)
        return node_id * self.gpus_per_node + local_rank

    def parse_files(self, log_files):
        """Parse múltiples archivos de log"""
        if not log_files:
            print("Error: No se proporcionaron archivos de log")
            return

        print(f"Parseando {len(log_files)} archivo(s)...")
        print(f"GPUs por nodo configuradas: {self.gpus_per_node}")
        print("="*70)

        for log_file in log_files:
            if not os.path.exists(log_file):
                print(f"⚠ Archivo no encontrado: {log_file}")
                continue

            self._parse_single_file(log_file)

        print()
        print("="*70)
        print(f"✓ Total de archivos procesados: {len(self.files_processed)}")
        print(f"✓ Total de nodos detectados: {len(self.hostname_to_node_id)}")
        print(f"✓ Operaciones colectivas: {len(self.communications)}")
        print(f"✓ Operaciones P2P: {len(self.p2p_communications)}")

        # Mostrar mapeo de hostnames
        print(f"\nMapeo hostname → node_id:")
        for hostname, node_id in sorted(self.hostname_to_node_id.items(), key=lambda x: x[1]):
            global_ranks = [node_id * self.gpus_per_node + i for i in range(self.gpus_per_node)]
            print(f"  {hostname} (node {node_id}) → global ranks {global_ranks[0]}-{global_ranks[-1]}")

        if self.parse_errors:
            print(f"\n⚠ Líneas con errores de parsing: {len(self.parse_errors)}")

    def _parse_single_file(self, log_file):
        """Parse un archivo de log individual"""
        basename = os.path.basename(log_file)
        print(f"  Procesando: {basename}")

        line_count = 0
        file_coll = 0
        file_p2p = 0
        first_timestamp_check = True

        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    line_count += 1

                    # Detectar si el log tiene timestamps
                    if first_timestamp_check and 'NCCL INFO' in line:
                        if line.strip().startswith('[') and ']' in line[:30]:
                            self.has_timestamps = True
                        first_timestamp_check = False

                    # Intentar parsear como operación colectiva
                    match = self.COLLECTIVE_PATTERN.search(line)
                    if match:
                        self._parse_collective(match, log_file, line_num, line)
                        file_coll += 1
                        continue

                    # Intentar parsear como operación P2P Send
                    match = self.P2P_SEND_PATTERN.search(line)
                    if match:
                        self._parse_p2p_send(match, log_file, line_num, line)
                        file_p2p += 1
                        continue

                    # Intentar parsear como operación P2P Recv
                    match = self.P2P_RECV_PATTERN.search(line)
                    if match:
                        self._parse_p2p_recv(match, log_file, line_num, line)
                        file_p2p += 1
                        continue

            self.files_processed.append(log_file)
            print(f"    → {line_count} líneas | {file_coll} colectivas | {file_p2p} P2P")

        except Exception as e:
            print(f"    ✗ Error procesando archivo: {e}")

    def _parse_collective(self, match, source_file, line_num, line):
        """Parse una operación colectiva"""
        try:
            timestamp, hostname, pid, tid, local_rank, operation, opcount, sendbuff, recvbuff, count, datatype, nranks = match.groups()

            local_rank = int(local_rank)
            count = int(count)
            datatype = int(datatype)
            nranks = int(nranks)

            # Calcular rank global
            global_rank = self._compute_global_rank(hostname, local_rank)

            # Calcular tamaño de datos en bytes
            element_size = self.DATATYPE_SIZES.get(datatype, 4)
            data_size_bytes = count * element_size

            comm_data = {
                'source_file': os.path.basename(source_file),
                'line': line_num,
                'hostname': hostname,
                'local_rank': local_rank,
                'global_rank': global_rank,
                'operation': operation,
                'opcount': opcount,
                'count': count,
                'datatype': datatype,
                'data_size_bytes': data_size_bytes,
                'data_size_mb': data_size_bytes / (1024 * 1024),
                'data_size_gb': data_size_bytes / (1024 * 1024 * 1024),
                'nranks': nranks,
                'sendbuff': sendbuff,
                'recvbuff': recvbuff,
            }

            # Agregar timestamp si existe
            if timestamp:
                comm_data['timestamp'] = timestamp
                try:
                    comm_data['datetime'] = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
                except:
                    pass

            self.communications.append(comm_data)
        except Exception as e:
            self.parse_errors.append((source_file, line_num))

    def _parse_p2p_send(self, match, source_file, line_num, line):
        """Parse una operación Send"""
        try:
            timestamp, hostname, pid, tid, local_rank, opcount, sendbuff, recvbuff, count, datatype, peer_local, nranks = match.groups()

            local_rank = int(local_rank)
            count = int(count)
            datatype = int(datatype)
            peer_local = int(peer_local)
            nranks = int(nranks)

            # Calcular ranks globales
            global_rank = self._compute_global_rank(hostname, local_rank)
            # Nota: peer podría ser en otro nodo, pero con la info actual asumimos mismo nodo
            # Para correcta conversión necesitaríamos más contexto

            # Calcular tamaño de datos
            element_size = self.DATATYPE_SIZES.get(datatype, 4)
            data_size_bytes = count * element_size

            p2p_data = {
                'source_file': os.path.basename(source_file),
                'line': line_num,
                'hostname': hostname,
                'local_rank': local_rank,
                'global_rank': global_rank,
                'operation': 'Send',
                'opcount': opcount,
                'peer_local': peer_local,
                'count': count,
                'datatype': datatype,
                'data_size_bytes': data_size_bytes,
                'data_size_mb': data_size_bytes / (1024 * 1024),
                'data_size_gb': data_size_bytes / (1024 * 1024 * 1024),
                'nranks': nranks,
                'buffer': sendbuff,
            }

            # Agregar timestamp si existe
            if timestamp:
                p2p_data['timestamp'] = timestamp
                try:
                    p2p_data['datetime'] = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
                except:
                    pass

            self.p2p_communications.append(p2p_data)
        except Exception as e:
            self.parse_errors.append((source_file, line_num))

    def _parse_p2p_recv(self, match, source_file, line_num, line):
        """Parse una operación Recv"""
        try:
            timestamp, hostname, pid, tid, local_rank, opcount, sendbuff, recvbuff, count, datatype, peer_local, nranks = match.groups()

            local_rank = int(local_rank)
            count = int(count)
            datatype = int(datatype)
            peer_local = int(peer_local)
            nranks = int(nranks)

            # Calcular ranks globales
            global_rank = self._compute_global_rank(hostname, local_rank)

            # Calcular tamaño de datos
            element_size = self.DATATYPE_SIZES.get(datatype, 4)
            data_size_bytes = count * element_size

            p2p_data = {
                'source_file': os.path.basename(source_file),
                'line': line_num,
                'hostname': hostname,
                'local_rank': local_rank,
                'global_rank': global_rank,
                'operation': 'Recv',
                'opcount': opcount,
                'peer_local': peer_local,
                'count': count,
                'datatype': datatype,
                'data_size_bytes': data_size_bytes,
                'data_size_mb': data_size_bytes / (1024 * 1024),
                'data_size_gb': data_size_bytes / (1024 * 1024 * 1024),
                'nranks': nranks,
                'buffer': recvbuff,
            }

            # Agregar timestamp si existe
            if timestamp:
                p2p_data['timestamp'] = timestamp
                try:
                    p2p_data['datetime'] = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
                except:
                    pass

            self.p2p_communications.append(p2p_data)
        except Exception as e:
            self.parse_errors.append((source_file, line_num))

    def analyze(self):
        """Analiza los datos parseados y genera estadísticas"""
        if not self.communications and not self.p2p_communications:
            print("No se encontraron comunicaciones NCCL en los logs")
            return None

        results = {}

        # Análisis de operaciones colectivas
        if self.communications:
            df_coll = pd.DataFrame(self.communications)
            results['collective'] = self._analyze_collective(df_coll)

        # Análisis de operaciones P2P
        if self.p2p_communications:
            df_p2p = pd.DataFrame(self.p2p_communications)
            results['p2p'] = self._analyze_p2p(df_p2p)

        return results

    def _analyze_collective(self, df):
        """Analiza operaciones colectivas"""
        analysis = {}

        # Resumen por tipo de operación
        op_summary = df.groupby('operation').agg({
            'operation': 'count',
            'data_size_mb': ['sum', 'mean', 'min', 'max'],
            'data_size_gb': 'sum'
        }).round(3)
        op_summary.columns = ['count', 'total_mb', 'avg_mb', 'min_mb', 'max_mb', 'total_gb']
        analysis['operation_summary'] = op_summary

        # Resumen por GLOBAL rank
        rank_summary = df.groupby('global_rank').agg({
            'operation': 'count',
            'data_size_mb': 'sum',
            'data_size_gb': 'sum',
            'hostname': 'first',
            'local_rank': 'first'
        }).round(3)
        rank_summary.columns = ['total_operations', 'total_data_mb', 'total_data_gb', 'hostname', 'local_rank']
        analysis['rank_summary'] = rank_summary

        # Resumen por hostname (nodo)
        node_summary = df.groupby('hostname').agg({
            'operation': 'count',
            'data_size_gb': 'sum',
            'global_rank': lambda x: list(sorted(x.unique()))
        }).round(3)
        node_summary.columns = ['operations', 'total_gb', 'global_ranks']
        analysis['node_summary'] = node_summary

        # Resumen por archivo fuente
        file_summary = df.groupby('source_file').agg({
            'operation': 'count',
            'data_size_gb': 'sum',
            'global_rank': 'first',
            'hostname': 'first'
        }).round(3)
        file_summary.columns = ['operations', 'total_gb', 'global_rank', 'hostname']
        analysis['file_summary'] = file_summary

        # Operaciones por global rank y tipo
        rank_op_matrix = pd.crosstab(df['global_rank'], df['operation'])
        analysis['rank_operation_matrix'] = rank_op_matrix

        # Top operaciones por tamaño de datos
        cols = ['source_file', 'hostname', 'global_rank', 'local_rank', 'operation', 'count', 'data_size_mb', 'opcount', 'line']
        if 'timestamp' in df.columns:
            cols.insert(0, 'timestamp')
        top_operations = df.nlargest(20, 'data_size_bytes')[cols]
        analysis['top_data_operations'] = top_operations

        # Estadísticas globales
        analysis['global_stats'] = {
            'total_operations': len(df),
            'total_data_mb': df['data_size_mb'].sum(),
            'total_data_gb': df['data_size_gb'].sum(),
            'num_global_ranks': df['global_rank'].nunique(),
            'num_nodes': df['hostname'].nunique(),
            'num_files': df['source_file'].nunique(),
            'operations_per_global_rank': df.groupby('global_rank').size().to_dict(),
            'avg_data_per_operation_mb': df['data_size_mb'].mean(),
            'has_timestamps': 'timestamp' in df.columns
        }

        # Análisis temporal si hay timestamps
        if 'datetime' in df.columns:
            df_time = df[df['datetime'].notna()].copy()
            if len(df_time) > 0:
                df_time = df_time.sort_values('datetime')
                time_range = (df_time['datetime'].max() - df_time['datetime'].min()).total_seconds()
                analysis['global_stats']['time_range_seconds'] = time_range
                analysis['global_stats']['first_timestamp'] = str(df_time['datetime'].min())
                analysis['global_stats']['last_timestamp'] = str(df_time['datetime'].max())
                analysis['global_stats']['operations_per_second'] = len(df_time) / time_range if time_range > 0 else 0

        return analysis

    def _analyze_p2p(self, df):
        """Analiza operaciones point-to-point"""
        analysis = {}

        # Matriz de comunicación usando global ranks
        comm_matrix = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'data_mb': 0.0}))

        # Nota: peer_local necesitaría convertirse a global, pero requiere saber el nodo del peer
        # Por ahora usar global_rank como aproximación
        for _, row in df.iterrows():
            if row['operation'] == 'Send':
                src = row['global_rank']
                # peer_local no se puede convertir sin más contexto, usar como está
                dst = row['peer_local']  # Esto es incorrecto pero necesitamos más info
                comm_matrix[src][dst]['count'] += 1
                comm_matrix[src][dst]['data_mb'] += row['data_size_mb']

        # Convertir a DataFrame
        ranks = sorted(set(df['global_rank'].unique()))
        count_matrix = pd.DataFrame(0, index=ranks, columns=ranks)
        data_matrix = pd.DataFrame(0.0, index=ranks, columns=ranks)

        for src in comm_matrix:
            for dst in comm_matrix[src]:
                if dst in ranks:
                    count_matrix.loc[src, dst] = comm_matrix[src][dst]['count']
                    data_matrix.loc[src, dst] = round(comm_matrix[src][dst]['data_mb'], 3)

        analysis['communication_count_matrix'] = count_matrix
        analysis['communication_data_matrix_mb'] = data_matrix

        # Resumen de envíos y recepciones por global rank
        sends = df[df['operation'] == 'Send'].groupby('global_rank').agg({
            'operation': 'count',
            'data_size_mb': 'sum',
            'data_size_gb': 'sum',
            'hostname': 'first'
        })
        sends.columns = ['send_count', 'send_data_mb', 'send_data_gb', 'hostname']

        recvs = df[df['operation'] == 'Recv'].groupby('global_rank').agg({
            'operation': 'count',
            'data_size_mb': 'sum',
            'data_size_gb': 'sum',
            'hostname': 'first'
        })
        recvs.columns = ['recv_count', 'recv_data_mb', 'recv_data_gb', 'hostname']

        p2p_summary = pd.concat([sends, recvs], axis=1).fillna(0).round(3)
        # Consolidar hostname
        if 'hostname_x' in p2p_summary.columns:
            p2p_summary['hostname'] = p2p_summary['hostname_x'].combine_first(p2p_summary['hostname_y'])
            p2p_summary = p2p_summary.drop(['hostname_x', 'hostname_y'], axis=1)

        analysis['p2p_summary'] = p2p_summary

        # Pares de comunicación más activos
        pair_comm = df[df['operation'] == 'Send'].groupby(['global_rank', 'peer_local']).agg({
            'operation': 'count',
            'data_size_mb': 'sum',
            'data_size_gb': 'sum'
        }).round(3)
        pair_comm.columns = ['count', 'total_data_mb', 'total_data_gb']
        pair_comm = pair_comm.sort_values('count', ascending=False)
        analysis['top_communication_pairs'] = pair_comm.head(30)

        return analysis

    def save_results(self, results, output_prefix):
        """Guarda los resultados en archivos CSV"""
        if not results:
            return

        print(f"\nGuardando resultados con prefijo: {output_prefix}")

        # Guardar operaciones colectivas
        if 'collective' in results:
            coll = results['collective']

            # Resumen por operación
            coll['operation_summary'].to_csv(f"{output_prefix}_collective_ops_summary.csv")
            print(f"  ✓ {output_prefix}_collective_ops_summary.csv")

            # Resumen por global rank
            coll['rank_summary'].to_csv(f"{output_prefix}_collective_rank_summary.csv")
            print(f"  ✓ {output_prefix}_collective_rank_summary.csv")

            # Resumen por nodo
            coll['node_summary'].to_csv(f"{output_prefix}_collective_node_summary.csv")
            print(f"  ✓ {output_prefix}_collective_node_summary.csv")

            # Resumen por archivo
            coll['file_summary'].to_csv(f"{output_prefix}_collective_file_summary.csv")
            print(f"  ✓ {output_prefix}_collective_file_summary.csv")

            # Matriz global rank-operación
            coll['rank_operation_matrix'].to_csv(f"{output_prefix}_collective_rank_op_matrix.csv")
            print(f"  ✓ {output_prefix}_collective_rank_op_matrix.csv")

            # Top operaciones
            coll['top_data_operations'].to_csv(f"{output_prefix}_collective_top_operations.csv", index=False)
            print(f"  ✓ {output_prefix}_collective_top_operations.csv")

        # Guardar operaciones P2P
        if 'p2p' in results:
            p2p = results['p2p']

            # Matrices de comunicación
            p2p['communication_count_matrix'].to_csv(f"{output_prefix}_p2p_comm_count_matrix.csv")
            print(f"  ✓ {output_prefix}_p2p_comm_count_matrix.csv")

            p2p['communication_data_matrix_mb'].to_csv(f"{output_prefix}_p2p_comm_data_matrix.csv")
            print(f"  ✓ {output_prefix}_p2p_comm_data_matrix.csv")

            # Resumen P2P por global rank
            p2p['p2p_summary'].to_csv(f"{output_prefix}_p2p_rank_summary.csv")
            print(f"  ✓ {output_prefix}_p2p_rank_summary.csv")

            # Pares más activos
            p2p['top_communication_pairs'].to_csv(f"{output_prefix}_p2p_top_pairs.csv")
            print(f"  ✓ {output_prefix}_p2p_top_pairs.csv")

        # Guardar todas las comunicaciones parseadas
        if self.communications:
            df_coll = pd.DataFrame(self.communications)
            df_coll.to_csv(f"{output_prefix}_all_collective_ops.csv", index=False)
            print(f"  ✓ {output_prefix}_all_collective_ops.csv")

        if self.p2p_communications:
            df_p2p = pd.DataFrame(self.p2p_communications)
            df_p2p.to_csv(f"{output_prefix}_all_p2p_ops.csv", index=False)
            print(f"  ✓ {output_prefix}_all_p2p_ops.csv")

    def print_summary(self, results):
        """Imprime un resumen de los resultados"""
        if not results:
            return

        print("\n" + "="*80)
        print("RESUMEN DE ANÁLISIS DE LOGS NCCL")
        print("="*80)

        if 'collective' in results:
            coll = results['collective']
            stats = coll['global_stats']

            print("\n### OPERACIONES COLECTIVAS ###")
            print(f"Total de operaciones: {stats['total_operations']:,}")
            print(f"Número de global ranks: {stats['num_global_ranks']} (GPUs únicas)")
            print(f"Número de nodos: {stats['num_nodes']}")
            print(f"Número de archivos: {stats['num_files']}")
            print(f"Datos totales transferidos: {stats['total_data_gb']:.2f} GB ({stats['total_data_mb']:.2f} MB)")
            print(f"Datos promedio por operación: {stats['avg_data_per_operation_mb']:.2f} MB")

            if stats.get('has_timestamps'):
                print(f"\n### INFORMACIÓN TEMPORAL ###")
                print(f"Primer timestamp: {stats.get('first_timestamp')}")
                print(f"Último timestamp: {stats.get('last_timestamp')}")
                print(f"Duración total: {stats.get('time_range_seconds', 0):.2f} segundos")
                print(f"Operaciones por segundo: {stats.get('operations_per_second', 0):.2f} ops/s")

            print(f"\nOperaciones por tipo:")
            print(coll['operation_summary'])

            print(f"\nOperaciones por nodo:")
            print(coll['node_summary'])

            print(f"\nOperaciones por global rank (Top 10):")
            print(coll['rank_summary'].head(10))

        if 'p2p' in results:
            p2p = results['p2p']

            print("\n### OPERACIONES POINT-TO-POINT ###")
            print(f"\nResumen de Send/Recv por global rank (Top 10):")
            print(p2p['p2p_summary'].head(10))


def main():
    parser = argparse.ArgumentParser(
        description='Parse y analiza logs de NCCL DEBUG - Multi-file con global rank',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Parsear todos los logs
  python nccl_parser.py ../tests/nccl_debug_* -o ../tests/results

  # Especificar GPUs por nodo (default: 4)
  python nccl_parser.py logs/*.log -o analysis --gpus-per-node 8

El parser calcula el rank global como:
  global_rank = node_id * gpus_per_node + local_rank

Ejemplo con 8 nodos y 4 GPUs por nodo:
  - as07r3b19 (node 0) + local_rank 0 → global_rank 0
  - as07r3b19 (node 0) + local_rank 1 → global_rank 1
  - as07r3b21 (node 1) + local_rank 0 → global_rank 4
  - as07r3b21 (node 1) + local_rank 1 → global_rank 5
  ...
  - Total: 32 global ranks (0-31)
        """
    )

    parser.add_argument('log_files', nargs='+', help='Archivo(s) de log NCCL a parsear')
    parser.add_argument('--output', '-o', default='nccl_analysis',
                       help='Prefijo para archivos de salida (default: nccl_analysis)')
    parser.add_argument('--gpus-per-node', '-g', type=int, default=4,
                       help='Número de GPUs por nodo (default: 4)')

    args = parser.parse_args()

    # Expandir globs si es necesario
    log_files = []
    for pattern in args.log_files:
        if '*' in pattern or '?' in pattern:
            expanded = glob.glob(pattern)
            if expanded:
                log_files.extend(expanded)
            else:
                print(f"⚠ No se encontraron archivos para el patrón: {pattern}")
        else:
            log_files.append(pattern)

    if not log_files:
        print("Error: No se encontraron archivos de log")
        sys.exit(1)

    # Crear parser y procesar
    nccl_parser = NCCLLogParser(gpus_per_node=args.gpus_per_node)
    nccl_parser.parse_files(log_files)

    # Analizar
    results = nccl_parser.analyze()

    # Mostrar resumen
    nccl_parser.print_summary(results)

    # Guardar resultados
    nccl_parser.save_results(results, args.output)

    print("\n✓ Análisis completado")


if __name__ == '__main__':
    main()