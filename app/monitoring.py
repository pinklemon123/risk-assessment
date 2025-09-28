from typing import Dict, List, Optional
import time
import logging
from dataclasses import dataclass
from enum import Enum

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class Alert:
    level: AlertLevel
    title: str
    message: str
    timestamp: float
    source: str
    tags: Dict[str, str]
    metric_value: Optional[float] = None
    threshold: Optional[float] = None

class MonitoringSystem:
    """监控告警系统"""
    
    def __init__(self):
        self.metrics_store = {}
        self.alert_rules = {}
        self.alert_channels = {}
        self.logger = logging.getLogger(__name__)
        
    def setup_monitoring_rules(self):
        """设置监控规则"""
        
        # 性能监控规则
        self.alert_rules.update({
            'api_response_time': {
                'metric': 'response_time_p95',
                'threshold': 5000,  # 5秒
                'comparison': 'greater_than',
                'level': AlertLevel.WARNING,
                'cooldown': 300  # 5分钟冷却
            },
            
            'api_error_rate': {
                'metric': 'error_rate_5m',
                'threshold': 0.05,  # 5%
                'comparison': 'greater_than',
                'level': AlertLevel.CRITICAL,
                'cooldown': 180
            },
            
            'queue_depth': {
                'metric': 'kafka_consumer_lag',
                'threshold': 10000,
                'comparison': 'greater_than',
                'level': AlertLevel.WARNING,
                'cooldown': 600
            }
        })
        
        # 业务监控规则
        self.alert_rules.update({
            'high_risk_transaction_rate': {
                'metric': 'high_risk_txn_rate_1h',
                'threshold': 0.20,  # 20%
                'comparison': 'greater_than',
                'level': AlertLevel.WARNING,
                'cooldown': 1800
            },
            
            'model_drift_detection': {
                'metric': 'model_performance_score',
                'threshold': 0.75,
                'comparison': 'less_than',
                'level': AlertLevel.CRITICAL,
                'cooldown': 3600
            },
            
            'feature_anomaly_count': {
                'metric': 'feature_anomaly_rate',
                'threshold': 0.10,  # 10%
                'comparison': 'greater_than', 
                'level': AlertLevel.WARNING,
                'cooldown': 900
            }
        })
        
        # 系统健康监控
        self.alert_rules.update({
            'cpu_utilization': {
                'metric': 'cpu_usage_avg_5m',
                'threshold': 80.0,  # 80%
                'comparison': 'greater_than',
                'level': AlertLevel.WARNING,
                'cooldown': 300
            },
            
            'memory_utilization': {
                'metric': 'memory_usage_percent',
                'threshold': 85.0,  # 85%
                'comparison': 'greater_than',
                'level': AlertLevel.CRITICAL,
                'cooldown': 300
            },
            
            'disk_space': {
                'metric': 'disk_usage_percent',
                'threshold': 90.0,  # 90%
                'comparison': 'greater_than',
                'level': AlertLevel.CRITICAL,
                'cooldown': 1800
            }
        })
    
    def setup_alert_channels(self):
        """设置告警渠道"""
        self.alert_channels = {
            'slack': {
                'webhook_url': '${SLACK_WEBHOOK_URL}',
                'channels': {
                    AlertLevel.INFO: '#fraud-detection-info',
                    AlertLevel.WARNING: '#fraud-detection-alerts', 
                    AlertLevel.CRITICAL: '#fraud-detection-critical',
                    AlertLevel.EMERGENCY: '#fraud-detection-emergency'
                }
            },
            
            'email': {
                'smtp_server': '${SMTP_SERVER}',
                'smtp_port': 587,
                'username': '${EMAIL_USERNAME}',
                'password': '${EMAIL_PASSWORD}',
                'recipients': {
                    AlertLevel.WARNING: ['team@company.com'],
                    AlertLevel.CRITICAL: ['team@company.com', 'oncall@company.com'],
                    AlertLevel.EMERGENCY: ['team@company.com', 'oncall@company.com', 'management@company.com']
                }
            },
            
            'pagerduty': {
                'api_key': '${PAGERDUTY_API_KEY}',
                'service_id': '${PAGERDUTY_SERVICE_ID}',
                'severity_mapping': {
                    AlertLevel.WARNING: 'warning',
                    AlertLevel.CRITICAL: 'error',
                    AlertLevel.EMERGENCY: 'critical'
                }
            }
        }
    
    def record_metric(self, metric_name: str, value: float, tags: Dict[str, str] = None):
        """记录监控指标"""
        timestamp = time.time()
        
        if metric_name not in self.metrics_store:
            self.metrics_store[metric_name] = []
            
        self.metrics_store[metric_name].append({
            'value': value,
            'timestamp': timestamp,
            'tags': tags or {}
        })
        
        # 检查是否触发告警
        self._check_alert_rules(metric_name, value, tags)
    
    def _check_alert_rules(self, metric_name: str, value: float, tags: Dict[str, str]):
        """检查告警规则"""
        for rule_name, rule_config in self.alert_rules.items():
            if rule_config['metric'] == metric_name:
                
                threshold = rule_config['threshold']
                comparison = rule_config['comparison']
                
                should_alert = False
                if comparison == 'greater_than' and value > threshold:
                    should_alert = True
                elif comparison == 'less_than' and value < threshold:
                    should_alert = True
                elif comparison == 'equals' and abs(value - threshold) < 0.001:
                    should_alert = True
                
                if should_alert:
                    alert = Alert(
                        level=rule_config['level'],
                        title=f"Alert: {rule_name}",
                        message=f"Metric {metric_name} = {value} {comparison} {threshold}",
                        timestamp=time.time(),
                        source="fraud_detection_system",
                        tags=tags or {},
                        metric_value=value,
                        threshold=threshold
                    )
                    
                    self._send_alert(alert)
    
    def _send_alert(self, alert: Alert):
        """发送告警"""
        self.logger.warning(f"Alert triggered: {alert.title} - {alert.message}")
        
        # 发送到Slack
        if 'slack' in self.alert_channels:
            self._send_slack_alert(alert)
            
        # 发送邮件
        if alert.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
            self._send_email_alert(alert)
            
        # 发送到PagerDuty
        if alert.level == AlertLevel.EMERGENCY:
            self._send_pagerduty_alert(alert)
    
    def get_system_health_dashboard(self) -> Dict:
        """获取系统健康面板数据"""
        return {
            'api_metrics': {
                'requests_per_second': self._get_metric_value('api_requests_per_second'),
                'average_response_time': self._get_metric_value('response_time_avg'),
                'error_rate': self._get_metric_value('error_rate_5m')
            },
            
            'business_metrics': {
                'transactions_processed': self._get_metric_value('transactions_processed_1h'),
                'high_risk_rate': self._get_metric_value('high_risk_txn_rate_1h'),
                'model_accuracy': self._get_metric_value('model_accuracy_score')
            },
            
            'system_metrics': {
                'cpu_usage': self._get_metric_value('cpu_usage_avg_5m'),
                'memory_usage': self._get_metric_value('memory_usage_percent'),
                'disk_usage': self._get_metric_value('disk_usage_percent')
            },
            
            'data_pipeline': {
                'kafka_consumer_lag': self._get_metric_value('kafka_consumer_lag'),
                'processing_latency': self._get_metric_value('processing_latency_p95'),
                'data_quality_score': self._get_metric_value('data_quality_score')
            }
        }
    
    def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """获取最新的指标值"""
        if metric_name in self.metrics_store and self.metrics_store[metric_name]:
            return self.metrics_store[metric_name][-1]['value']
        return None

# 监控配置YAML
MONITORING_CONFIG = """
monitoring:
  collection_interval: 30  # 秒
  
  metrics:
    # API性能指标
    api_response_time:
      type: histogram
      buckets: [0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
      
    api_requests_total:
      type: counter
      labels: [method, endpoint, status]
      
    # 业务指标
    transactions_processed:
      type: counter
      labels: [status, risk_level]
      
    fraud_score_distribution:
      type: histogram  
      buckets: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
      
    # 系统指标
    system_cpu_usage:
      type: gauge
      
    system_memory_usage:
      type: gauge
      
    kafka_consumer_lag:
      type: gauge
      labels: [topic, partition]

  alerts:
    channels:
      - name: slack
        type: webhook
        config:
          url: ${SLACK_WEBHOOK_URL}
          
      - name: email 
        type: smtp
        config:
          host: ${SMTP_HOST}
          port: 587
          username: ${SMTP_USER}
          password: ${SMTP_PASS}
          
    rules:
      - name: high_api_latency
        condition: api_response_time_p95 > 5
        severity: warning
        channels: [slack]
        
      - name: high_error_rate
        condition: api_error_rate_5m > 0.05
        severity: critical  
        channels: [slack, email]
        
      - name: model_degradation
        condition: model_accuracy < 0.8
        severity: critical
        channels: [slack, email]
"""