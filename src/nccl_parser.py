
#!/usr/bin/env python3
"""
NCCL Debug Log Parser and Analyzer
Parsea logs de NCCL generados con NCCL_DEBUG=INFO
Soporta formato con y sin timestamps:
  - Con timestamp: [2025-10-27 17:36:20.330]as02r3b05:2840695:2840918 [2] NCCL INFO ...
  - Sin timestamp: as05r3b03:1912985:1912985 [0] NCCL INFO ...

Uso:
    python nccl_parser.py <log_file> [--output <output_prefix>]
"""

import re
import sys
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime
import argparse
import json

class NCCLLogParser:
    """Parser para logs de NCCL DEBUG con timestamps opcionales"""

    # Patrón mejorado para operaciones colectivas (timestamp opcional)
    COLLECTIVE_PATTERN = re.compile(
        r'^(?:\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\])?'  # timestamp opcional
        r'\S+:\d+:\d+\s+'  # hostname:pid:tid
        r'\[(\d+)\]\s+'  # rank
        r'NCCL INFO\s+'
        r'(AllReduce|AllGather|Broadcast|ReduceScatter|Reduce|AllToAll):\s*'  # operation
        r'opCount\s+(\d+|[0-9a-fA-F]+)\s+'  # opCount
        r'sendbuff\s+(0x[0-9a-fA-F]+|\(nil\))\s+'  # sendbuff
        r'recvbuff\s+(0x[0-9a-fA-F]+|\(nil\))\s+'  # recvbuff
        r'count\s+(\d+)\s+'  # count
        r'datatype\s+(\d+)'  # datatype
        r'.*\[nranks=(\d+)\]'  # nranks
    )

    # Patrón mejorado para operaciones P2P Send (timestamp opcional)
    P2P_SEND_PATTERN = re.compile(
        r'^(?:\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\])?'  # timestamp opcional
        r'\S+:\d+:\d+\s+'  # hostname:pid:tid
        r'\[(\d+)\]\s+'  # rank
        r'NCCL INFO\s+'
        r'Send:\s*'  # operation
        r'opCount\s+(\d+|[0-9a-fA-F]+)\s+'  # opCount
        r'sendbuff\s+(0x[0-9a-fA-F]+|\(nil\))\s+'  # sendbuff (para Send)
        r'recvbuff\s+(0x[0-9a-fA-F]+|\(nil\))\s+'  # recvbuff
        r'count\s+(\d+)\s+'  # count
        r'datatype\s+(\d+)\s+'  # datatype
        r'.*root\s+(\d+)'  # peer rank (en root field)
        r'.*\[nranks=(\d+)\]'  # nranks
    )

    # Patrón mejorado para operaciones P2P Recv (timestamp opcional)
    P2P_RECV_PATTERN = re.compile(
        r'^(?:\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\])?'  # timestamp opcional
        r'\S+:\d+:\d+\s+'  # hostname:pid:tid
        r'\[(\d+)\]\s+'  # rank
        r'NCCL INFO\s+'
        r'Recv:\s*'  # operation
        r'opCount\s+(\d+|[0-9a-fA-F]+)\s+'  # opCount
        r'sendbuff\s+(0x[0-9a-fA-F]+|\(nil\))\s+'  # sendbuff
        r'recvbuff\s+(0x[0-9a-fA-F]+|\(nil\))\s+'  # recvbuff
        r'count\s+(\d+)\s+'  # count
        r'datatype\s+(\d+)\s+'  # datatype
        r'.*root\s+(\d+)'  # peer rank (en root field)
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

    def __init__(self):
        self.communications = []
        self.p2p_communications = []
        self.parse_errors = []
        self.has_timestamps = False

    def parse_file(self, log_file):
        """Parse el archivo de log completo"""
        print(f"Parseando archivo: {log_file}")

        line_count = 0
        first_timestamp_check = True

        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line_count += 1

                # Detectar si el log tiene timestamps (solo en primera operación)
                if first_timestamp_check and 'NCCL INFO' in line:
                    if line.strip().startswith('[') and ']' in line[:30]:
                        self.has_timestamps = True
                        print("✓ Detectados timestamps en el log")
                    first_timestamp_check = False

                # Intentar parsear como operación colectiva
                match = self.COLLECTIVE_PATTERN.search(line)
                if match:
                    self._parse_collective(match, line_num, line)
                    continue

                # Intentar parsear como operación P2P Send
                match = self.P2P_SEND_PATTERN.search(line)
                if match:
                    self._parse_p2p_send(match, line_num, line)
                    continue

                # Intentar parsear como operación P2P Recv
                match = self.P2P_RECV_PATTERN.search(line)
                if match:
                    self._parse_p2p_recv(match, line_num, line)
                    continue

        print(f"Líneas totales procesadas: {line_count}")
        print(f"Encontradas {len(self.communications)} operaciones colectivas")
        print(f"Encontradas {len(self.p2p_communications)} operaciones P2P")

        if self.parse_errors:
            print(f"\nAdvertencia: {len(self.parse_errors)} líneas con errores de parsing")
            print("Primeros 5 errores:")
            for err in self.parse_errors[:5]:
                print(f"  Línea {err}")

    def _parse_collective(self, match, line_num, line):
        """Parse una operación colectiva"""
        try:
            timestamp, rank, operation, opcount, sendbuff, recvbuff, count, datatype, nranks = match.groups()

            rank = int(rank)
            count = int(count)
            datatype = int(datatype)
            nranks = int(nranks)

            # Calcular tamaño de datos en bytes
            element_size = self.DATATYPE_SIZES.get(datatype, 4)
            data_size_bytes = count * element_size

            comm_data = {
                'line': line_num,
                'rank': rank,
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
            self.parse_errors.append(line_num)

    def _parse_p2p_send(self, match, line_num, line):
        """Parse una operación Send"""
        try:
            timestamp, rank, opcount, sendbuff, recvbuff, count, datatype, peer, nranks = match.groups()

            rank = int(rank)
            count = int(count)
            datatype = int(datatype)
            peer = int(peer)
            nranks = int(nranks)

            # Calcular tamaño de datos
            element_size = self.DATATYPE_SIZES.get(datatype, 4)
            data_size_bytes = count * element_size

            p2p_data = {
                'line': line_num,
                'rank': rank,
                'operation': 'Send',
                'opcount': opcount,
                'peer': peer,
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
            self.parse_errors.append(line_num)

    def _parse_p2p_recv(self, match, line_num, line):
        """Parse una operación Recv"""
        try:
            timestamp, rank, opcount, sendbuff, recvbuff, count, datatype, peer, nranks = match.groups()

            rank = int(rank)
            count = int(count)
            datatype = int(datatype)
            peer = int(peer)
            nranks = int(nranks)

            # Calcular tamaño de datos
            element_size = self.DATATYPE_SIZES.get(datatype, 4)
            data_size_bytes = count * element_size

            p2p_data = {
                'line': line_num,
                'rank': rank,
                'operation': 'Recv',
                'opcount': opcount,
                'peer': peer,
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
            self.parse_errors.append(line_num)

    def analyze(self):
        """Analiza los datos parseados y genera estadísticas"""
        if not self.communications and not self.p2p_communications:
            print("No se encontraron comunicaciones NCCL en el log")
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

        # Resumen por rank
        rank_summary = df.groupby('rank').agg({
            'operation': 'count',
            'data_size_mb': 'sum',
            'data_size_gb': 'sum'
        }).round(3)
        rank_summary.columns = ['total_operations', 'total_data_mb', 'total_data_gb']
        analysis['rank_summary'] = rank_summary

        # Operaciones por rank y tipo
        rank_op_matrix = pd.crosstab(df['rank'], df['operation'])
        analysis['rank_operation_matrix'] = rank_op_matrix

        # Top operaciones por tamaño de datos
        cols = ['rank', 'operation', 'count', 'data_size_mb', 'opcount', 'line']
        if 'timestamp' in df.columns:
            cols.insert(0, 'timestamp')
        top_operations = df.nlargest(20, 'data_size_bytes')[cols]
        analysis['top_data_operations'] = top_operations

        # Estadísticas globales
        analysis['global_stats'] = {
            'total_operations': len(df),
            'total_data_mb': df['data_size_mb'].sum(),
            'total_data_gb': df['data_size_gb'].sum(),
            'num_ranks': df['rank'].nunique(),
            'operations_per_rank': df.groupby('rank').size().to_dict(),
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

        # Matriz de comunicación (quién envía a quién)
        comm_matrix = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'data_mb': 0.0}))

        for _, row in df.iterrows():
            if row['operation'] == 'Send':
                src, dst = row['rank'], row['peer']
                comm_matrix[src][dst]['count'] += 1
                comm_matrix[src][dst]['data_mb'] += row['data_size_mb']

        # Convertir a DataFrame
        ranks = sorted(set(df['rank'].unique()) | set(df['peer'].unique()))
        count_matrix = pd.DataFrame(0, index=ranks, columns=ranks)
        data_matrix = pd.DataFrame(0.0, index=ranks, columns=ranks)

        for src in comm_matrix:
            for dst in comm_matrix[src]:
                count_matrix.loc[src, dst] = comm_matrix[src][dst]['count']
                data_matrix.loc[src, dst] = round(comm_matrix[src][dst]['data_mb'], 3)

        analysis['communication_count_matrix'] = count_matrix
        analysis['communication_data_matrix_mb'] = data_matrix

        # Resumen de envíos y recepciones por rank
        sends = df[df['operation'] == 'Send'].groupby('rank').agg({
            'operation': 'count',
            'data_size_mb': 'sum',
            'data_size_gb': 'sum'
        })
        sends.columns = ['send_count', 'send_data_mb', 'send_data_gb']

        recvs = df[df['operation'] == 'Recv'].groupby('rank').agg({
            'operation': 'count',
            'data_size_mb': 'sum',
            'data_size_gb': 'sum'
        })
        recvs.columns = ['recv_count', 'recv_data_mb', 'recv_data_gb']

        p2p_summary = pd.concat([sends, recvs], axis=1).fillna(0).round(3)
        analysis['p2p_summary'] = p2p_summary

        # Pares de comunicación más activos
        pair_comm = df[df['operation'] == 'Send'].groupby(['rank', 'peer']).agg({
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

            # Resumen por rank
            coll['rank_summary'].to_csv(f"{output_prefix}_collective_rank_summary.csv")
            print(f"  ✓ {output_prefix}_collective_rank_summary.csv")

            # Matriz rank-operación
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

            # Resumen P2P por rank
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
            print(f"Total de operaciones: {stats['total_operations']}")
            print(f"Número de ranks: {stats['num_ranks']}")
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
            print(f"\nOperaciones por rank (Top 10):")
            print(coll['rank_summary'].head(10))

        if 'p2p' in results:
            p2p = results['p2p']

            print("\n### OPERACIONES POINT-TO-POINT ###")
            print(f"\nResumen de Send/Recv por rank (Top 10):")
            print(p2p['p2p_summary'].head(10))
            print(f"\nTop 10 pares de comunicación más activos:")
            print(p2p['top_communication_pairs'].head(10))


def main():
    parser = argparse.ArgumentParser(
        description='Parse y analiza logs de NCCL DEBUG - Versión 3 con timestamps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python nccl_parser_v3.py nccl_debug.log
  python nccl_parser_v3.py nccl_debug.log --output results/analysis

Para generar logs NCCL:
  # Sin timestamps
  NCCL_DEBUG=INFO <tu_comando> 2>&1 | tee nccl_debug.log

  # Con timestamps
  NCCL_DEBUG=INFO <tu_comando> 2>&1 | ts '[%Y-%m-%d %H:%M:%.S]' | tee nccl_debug.log
        """
    )

    parser.add_argument('log_file', help='Archivo de log NCCL a parsear')
    parser.add_argument('--output', '-o', default='nccl_analysis',
                       help='Prefijo para archivos de salida (default: nccl_analysis)')

    args = parser.parse_args()

    # Crear parser y procesar
    nccl_parser = NCCLLogParser()
    nccl_parser.parse_file(args.log_file)

    # Analizar
    results = nccl_parser.analyze()

    # Mostrar resumen
    nccl_parser.print_summary(results)

    # Guardar resultados
    nccl_parser.save_results(results, args.output)

    print("\n✓ Análisis completado")


if __name__ == '__main__':
    main()
