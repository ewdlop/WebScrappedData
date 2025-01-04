import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

class TrafficPatternClassifier:
    def __init__(self):
        self.logger = self._setup_logger()
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(random_state=42, contamination=0.1)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def extract_features(self, logs: List[Dict]) -> pd.DataFrame:
        """
        Extract relevant features from traffic logs for analysis.
        
        Args:
            logs: List of dictionaries containing traffic data
            
        Returns:
            DataFrame with extracted features
        """
        features = []
        for ip in set(log['ip'] for log in logs):
            ip_logs = [log for log in logs if log['ip'] == ip]
            
            # Time-based features
            timestamps = sorted([log['timestamp'] for log in ip_logs])
            time_diffs = np.diff([ts.timestamp() for ts in timestamps])
            
            # Request pattern features
            requests = [log['request'] for log in ip_logs]
            unique_paths = len(set(req['path'] for req in requests))
            
            # Calculate features
            feature = {
                'ip': ip,
                'request_count': len(ip_logs),
                'avg_time_between_requests': np.mean(time_diffs) if len(time_diffs) > 0 else 0,
                'std_time_between_requests': np.std(time_diffs) if len(time_diffs) > 0 else 0,
                'unique_paths_ratio': unique_paths / len(ip_logs),
                'success_ratio': sum(1 for log in ip_logs if log['status_code'] < 400) / len(ip_logs),
                'error_ratio': sum(1 for log in ip_logs if log['status_code'] >= 400) / len(ip_logs),
                'bytes_transferred_mean': np.mean([log['bytes'] for log in ip_logs]),
                'user_agent_count': len(set(log['user_agent'] for log in ip_logs))
            }
            
            # Add time window features
            time_windows = [1, 5, 15, 30, 60]  # minutes
            for window in time_windows:
                count = self._count_requests_in_window(timestamps, window)
                feature[f'requests_{window}m'] = count
            
            features.append(feature)
        
        return pd.DataFrame(features)

    def _count_requests_in_window(self, timestamps: List[datetime], 
                                window_minutes: int) -> int:
        """Count max requests within sliding time window."""
        if not timestamps:
            return 0
            
        max_count = 0
        window = timedelta(minutes=window_minutes)
        
        for i in range(len(timestamps)):
            window_end = timestamps[i]
            window_start = window_end - window
            count = sum(1 for ts in timestamps if window_start <= ts <= window_end)
            max_count = max(max_count, count)
            
        return max_count

    def analyze_patterns(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Analyze traffic patterns using multiple detection methods.
        
        Args:
            features_df: DataFrame with extracted features
            
        Returns:
            Tuple of (DataFrame with results, Dict with analysis metadata)
        """
        # Prepare features for modeling
        feature_cols = [col for col in features_df.columns 
                       if col not in ['ip'] and features_df[col].dtype in ['int64', 'float64']]
        X = features_df[feature_cols]
        X_scaled = self.scaler.fit_transform(X)
        
        # Run Isolation Forest
        if_predictions = self.isolation_forest.fit_predict(X_scaled)
        
        # Run DBSCAN
        dbscan_labels = self.dbscan.fit_predict(X_scaled)
        
        # Combine results
        results_df = features_df.copy()
        results_df['anomaly_score'] = if_predictions
        results_df['cluster_label'] = dbscan_labels
        
        # Calculate additional metrics
        metrics = {
            'total_ips': len(features_df),
            'anomaly_count': sum(if_predictions == -1),
            'cluster_count': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
            'noise_points': sum(dbscan_labels == -1),
            'timestamp': datetime.now()
        }
        
        return results_df, metrics

    def get_detailed_analysis(self, results_df: pd.DataFrame) -> Dict:
        """
        Generate detailed analysis of classification results.
        
        Args:
            results_df: DataFrame with classification results
            
        Returns:
            Dictionary containing detailed analysis
        """
        analysis = {
            'high_frequency_ips': self._get_high_frequency_ips(results_df),
            'pattern_clusters': self._analyze_clusters(results_df),
            'anomaly_patterns': self._analyze_anomalies(results_df),
            'recommendations': self._generate_recommendations(results_df)
        }
        return analysis

    def _get_high_frequency_ips(self, df: pd.DataFrame) -> List[Dict]:
        """Identify IPs with unusually high request frequencies."""
        threshold = df['request_count'].mean() + 2 * df['request_count'].std()
        high_freq = df[df['request_count'] > threshold]
        return high_freq.to_dict('records')

    def _analyze_clusters(self, df: pd.DataFrame) -> List[Dict]:
        """Analyze characteristics of identified traffic clusters."""
        clusters = []
        for label in sorted(df['cluster_label'].unique()):
            if label != -1:  # Skip noise points
                cluster_data = df[df['cluster_label'] == label]
                cluster_info = {
                    'cluster_id': label,
                    'size': len(cluster_data),
                    'avg_request_count': cluster_data['request_count'].mean(),
                    'avg_time_between_requests': cluster_data['avg_time_between_requests'].mean(),
                    'common_patterns': self._identify_common_patterns(cluster_data)
                }
                clusters.append(cluster_info)
        return clusters

    def _analyze_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Analyze characteristics of anomalous traffic."""
        anomalies = df[df['anomaly_score'] == -1]
        analysis = []
        for _, anomaly in anomalies.iterrows():
            analysis.append({
                'ip': anomaly['ip'],
                'request_count': anomaly['request_count'],
                'unusual_patterns': self._identify_unusual_patterns(anomaly),
                'risk_level': self._calculate_risk_level(anomaly)
            })
        return analysis

    def _identify_common_patterns(self, cluster_data: pd.DataFrame) -> List[str]:
        """Identify common patterns within a cluster."""
        patterns = []
        
        # Request frequency patterns
        avg_req_count = cluster_data['request_count'].mean()
        if avg_req_count > cluster_data['request_count'].median() * 2:
            patterns.append('High request frequency variation')
        
        # Time patterns
        if cluster_data['avg_time_between_requests'].std() < 0.1:
            patterns.append('Consistent request timing')
        
        # Path patterns
        if cluster_data['unique_paths_ratio'].mean() < 0.2:
            patterns.append('Limited path variety')
            
        return patterns

    def _identify_unusual_patterns(self, row: pd.Series) -> List[str]:
        """Identify unusual patterns for an anomalous IP."""
        patterns = []
        
        if row['request_count'] > 1000:
            patterns.append('Extremely high request count')
            
        if row['avg_time_between_requests'] < 0.1:
            patterns.append('Unusually rapid requests')
            
        if row['error_ratio'] > 0.5:
            patterns.append('High error rate')
            
        if row['user_agent_count'] > 10:
            patterns.append('Multiple user agents')
            
        return patterns

    def _calculate_risk_level(self, row: pd.Series) -> str:
        """Calculate risk level based on various metrics."""
        risk_score = 0
        
        # Add risk points based on different factors
        if row['request_count'] > 1000:
            risk_score += 3
        elif row['request_count'] > 500:
            risk_score += 2
            
        if row['avg_time_between_requests'] < 0.1:
            risk_score += 3
            
        if row['error_ratio'] > 0.5:
            risk_score += 2
            
        if row['user_agent_count'] > 10:
            risk_score += 2
            
        # Determine risk level
        if risk_score >= 8:
            return 'High'
        elif risk_score >= 5:
            return 'Medium'
        else:
            return 'Low'

    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # Check for high-frequency request patterns
        if (df['request_count'] > 1000).any():
            recommendations.append(
                'Consider implementing rate limiting for high-frequency requestors'
            )
            
        # Check for multiple user agents
        if (df['user_agent_count'] > 10).any():
            recommendations.append(
                'Monitor IPs using multiple user agents'
            )
            
        # Check for error rates
        if (df['error_ratio'] > 0.5).any():
            recommendations.append(
                'Investigate IPs with high error rates'
            )
            
        # Check for rapid requests
        if (df['avg_time_between_requests'] < 0.1).any():
            recommendations.append(
                'Implement request timing analysis'
            )
            
        return recommendations

# Example usage
if __name__ == "__main__":
    # Create sample traffic logs
    sample_logs = [
        {
            'ip': '192.168.1.1',
            'timestamp': datetime.now() - timedelta(minutes=5),
            'request': {'path': '/page1'},
            'status_code': 200,
            'bytes': 1024,
            'user_agent': 'Mozilla/5.0'
        },
        # Add more sample logs...
    ]
    
    # Initialize classifier
    classifier = TrafficPatternClassifier()
    
    # Extract features
    features_df = classifier.extract_features(sample_logs)
    
    # Analyze patterns
    results_df, metrics = classifier.analyze_patterns(features_df)
    
    # Get detailed analysis
    analysis = classifier.get_detailed_analysis(results_df)
    
    # Print results
    print("Analysis Metrics:", metrics)
    print("\nDetailed Analysis:", analysis)
