#!/usr/bin/env python3
"""
大规模真实数据生成器 - 生成需要大量计算的复杂交易数据集
包含各种欺诈模式、网络关系、时间序列异常等需要深度分析的数据
"""

import json
import random
import time
from datetime import datetime, timedelta
import uuid
import math
import argparse

# 复杂的欺诈模式配置
FRAUD_PATTERNS = {
    "card_skimming": {
        "probability": 0.02,
        "characteristics": ["multiple_locations", "small_amounts", "frequent_transactions"]
    },
    "account_takeover": {
        "probability": 0.01,
        "characteristics": ["device_change", "location_change", "large_amounts"]
    },
    "money_laundering": {
        "probability": 0.005,
        "characteristics": ["circular_transfers", "cash_intensive", "rapid_movement"]
    },
    "synthetic_identity": {
        "probability": 0.008,
        "characteristics": ["new_account", "gradual_buildup", "mixed_patterns"]
    },
    "collusive_fraud": {
        "probability": 0.003,
        "characteristics": ["coordinated_timing", "shared_merchants", "similar_amounts"]
    }
}

# 真实商户类别及其风险权重
MERCHANT_CATEGORIES = {
    "grocery": {"weight": 0.4, "risk_score": 0.1, "avg_amount": [20, 200]},
    "gas_station": {"weight": 0.3, "risk_score": 0.15, "avg_amount": [30, 100]},
    "restaurant": {"weight": 0.25, "risk_score": 0.2, "avg_amount": [15, 150]},
    "retail": {"weight": 0.2, "risk_score": 0.25, "avg_amount": [50, 500]},
    "online": {"weight": 0.15, "risk_score": 0.4, "avg_amount": [25, 1000]},
    "atm_cash": {"weight": 0.1, "risk_score": 0.3, "avg_amount": [100, 500]},
    "crypto": {"weight": 0.02, "risk_score": 0.9, "avg_amount": [1000, 50000]},
    "casino": {"weight": 0.01, "risk_score": 0.95, "avg_amount": [500, 25000]},
    "jewelry": {"weight": 0.005, "risk_score": 0.7, "avg_amount": [1000, 10000]},
    "electronics": {"weight": 0.08, "risk_score": 0.35, "avg_amount": [200, 2000]}
}

# 地理位置数据
LOCATIONS = {
    "beijing": {"lat": 39.9042, "lon": 116.4074, "risk_multiplier": 1.0},
    "shanghai": {"lat": 31.2304, "lon": 121.4737, "risk_multiplier": 1.0},
    "shenzhen": {"lat": 22.5431, "lon": 114.0579, "risk_multiplier": 1.1},
    "guangzhou": {"lat": 23.1291, "lon": 113.2644, "risk_multiplier": 1.0},
    "hangzhou": {"lat": 30.2741, "lon": 120.1551, "risk_multiplier": 1.0},
    "macau": {"lat": 22.1987, "lon": 113.5439, "risk_multiplier": 1.8},
    "hong_kong": {"lat": 22.3193, "lon": 114.1694, "risk_multiplier": 1.4},
    "unknown": {"lat": 0, "lon": 0, "risk_multiplier": 2.0},
    "overseas": {"lat": random.uniform(-90, 90), "lon": random.uniform(-180, 180), "risk_multiplier": 1.6}
}

class ComplexDatasetGenerator:
    def __init__(self, *, compactness: float = 0.0, account_pool_size: int | None = None,
                 merchant_pool_size: int | None = None, reuse_prob: float | None = None,
                 id_prefix_account: str = "acc", id_prefix_merchant: str = "mer",
                 id_format_account: str | None = None, id_format_merchant: str | None = None):
        """复杂数据集生成器

        参数说明：
        - compactness: 0~1，网络紧凑度，越大代表越倾向于复用同一批账户/商户，形成更密的网络。
        - account_pool_size/merchant_pool_size: 限制可用的唯一账户/商户数量（若未提供，则由紧凑度推导）。
        - reuse_prob: 0~1，交易生成时复用现有账户/商户的概率（未提供则基于紧凑度自动推导）。
        """

        # 实例存储
        self.accounts = {}
        self.merchants = {}
        self.devices = {}
        self.transaction_networks = []
        self.fraud_rings = []

        # 紧凑模式配置
        self.compactness = max(0.0, min(1.0, compactness or 0.0))
        self.account_pool_size = account_pool_size
        self.merchant_pool_size = merchant_pool_size
        # 默认复用概率：随紧凑度线性上升，且不超过 0.95
        self.reuse_prob = min(0.95, max(0.0, reuse_prob if reuse_prob is not None else (0.6 + 0.35 * self.compactness)))

        # ID 命名配置
        self.id_prefix_account = (id_prefix_account or "acc").strip('-_')
        self.id_prefix_merchant = (id_prefix_merchant or "mer").strip('-_')
        # 支持形如 "{prefix}-{type}-{num:05d}" 的格式；若为空则默认 "{prefix}_pool_{num:05d}" / "{prefix}_{rand}"
        self.id_format_account = id_format_account
        self.id_format_merchant = id_format_merchant

        # 账户/商户池（在紧凑模式下启用）
        self.account_pool = []
        self.merchant_pool = []
        self._compact_mode_enabled = False
        
    def _maybe_build_pools(self, expected_txn_count: int):
        """根据紧凑度与期望交易量，构建账户与商户池。

        当 compactness>0 或显式给定池大小时启用紧凑模式。
        """
        if self.compactness <= 0 and not (self.account_pool_size or self.merchant_pool_size):
            # 非紧凑模式，无需构建
            self._compact_mode_enabled = False
            return

        # 根据紧凑度推导目标唯一实体数（保留合理下界）
        def derive_pool_size(default_ratio_high: float, subtract_ratio: float, min_abs: int) -> int:
            # size ≈ count * (default_ratio_high - subtract_ratio * compactness)
            size = int(expected_txn_count * (default_ratio_high - subtract_ratio * self.compactness))
            return max(min_abs, size)

        acct_size = self.account_pool_size or derive_pool_size(default_ratio_high=0.5, subtract_ratio=0.45, min_abs=50)
        mer_size = self.merchant_pool_size or derive_pool_size(default_ratio_high=0.8, subtract_ratio=0.7, min_abs=80)

        # 预先生成账户池
        for i in range(acct_size):
            if self.id_format_account:
                acc_id = self.id_format_account.format(prefix=self.id_prefix_account, type='C', num=i)
            else:
                acc_id = f"{self.id_prefix_account}_pool_{i:05d}"
            if acc_id not in self.accounts:
                self.generate_account_profile(acc_id)
            self.account_pool.append(acc_id)

        # 预先生成商户池（按类别权重采样）
        categories = list(MERCHANT_CATEGORIES.keys())
        weights = [MERCHANT_CATEGORIES[c]["weight"] for c in categories]
        for i in range(mer_size):
            cat = random.choices(categories, weights=weights)[0]
            if self.id_format_merchant:
                mer_id = self.id_format_merchant.format(prefix=self.id_prefix_merchant, type='S', num=i, cat=cat)
            else:
                mer_id = f"pool_{cat}_{i:05d}"
            if mer_id not in self.merchants:
                self.generate_merchant_profile(mer_id, cat)
            self.merchant_pool.append(mer_id)

        self._compact_mode_enabled = True

    def generate_account_profile(self, account_id):
        """生成复杂的账户画像"""
        age_days = random.randint(1, 3650)  # 1-10年账户历史
        
        # 基于账户年龄的行为模式
        if age_days < 30:
            transaction_frequency = random.uniform(0.5, 2.0)  # 新账户较少交易
            risk_baseline = 0.3
        elif age_days < 365:
            transaction_frequency = random.uniform(1.0, 5.0)
            risk_baseline = 0.15
        else:
            transaction_frequency = random.uniform(2.0, 8.0)  # 老账户更活跃
            risk_baseline = 0.1
            
        profile = {
            "account_id": account_id,
            "creation_date": datetime.now() - timedelta(days=age_days),
            "age_days": age_days,
            "transaction_frequency": transaction_frequency,
            "risk_baseline": risk_baseline,
            "preferred_locations": random.sample(list(LOCATIONS.keys()), random.randint(1, 3)),
            "preferred_merchants": random.sample(list(MERCHANT_CATEGORIES.keys()), random.randint(2, 5)),
            "credit_limit": random.uniform(5000, 100000),
            "monthly_income": random.uniform(3000, 50000),
            "is_fraudster": random.random() < 0.05,  # 5%的账户是潜在欺诈者
            "device_ids": [f"device_{uuid.uuid4().hex[:8]}" for _ in range(random.randint(1, 3))]
        }
        
        self.accounts[account_id] = profile
        return profile
        
    def generate_merchant_profile(self, merchant_id, category):
        """生成商户画像"""
        profile = {
            "merchant_id": merchant_id,
            "category": category,
            "risk_score": MERCHANT_CATEGORIES[category]["risk_score"],
            "location": random.choice(list(LOCATIONS.keys())),
            "avg_transaction_amount": random.uniform(*MERCHANT_CATEGORIES[category]["avg_amount"]),
            "is_suspicious": random.random() < 0.02,  # 2%的商户有问题
            "operating_hours": (random.randint(6, 10), random.randint(20, 24)),
            "fraud_history": random.random() < 0.01
        }
        
        self.merchants[merchant_id] = profile
        return profile
        
    def generate_fraud_ring(self, ring_size=random.randint(3, 8)):
        """生成欺诈团伙"""
        ring_accounts = [f"fraud_ring_acc_{uuid.uuid4().hex[:6]}" for _ in range(ring_size)]
        ring_merchants = [f"fraud_ring_mer_{uuid.uuid4().hex[:6]}" for _ in range(random.randint(1, 3))]
        
        # 生成团伙成员画像
        for acc in ring_accounts:
            profile = self.generate_account_profile(acc)
            profile["is_fraudster"] = True
            profile["fraud_ring_id"] = f"ring_{uuid.uuid4().hex[:8]}"
            
        ring = {
            "accounts": ring_accounts,
            "merchants": ring_merchants,
            "fraud_pattern": random.choice(list(FRAUD_PATTERNS.keys())),
            "coordination_level": random.uniform(0.3, 0.9)
        }
        
        self.fraud_rings.append(ring)
        return ring
        
    def calculate_geographic_risk(self, locations, time_diff_hours):
        """计算地理位置风险（基于距离和时间）"""
        if len(locations) < 2:
            return 0.0
            
        # 计算两点距离（简化的球面距离）
        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371  # 地球半径（公里）
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = (math.sin(dlat/2) * math.sin(dlat/2) + 
                 math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
                 math.sin(dlon/2) * math.sin(dlon/2))
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            return R * c
            
        loc1, loc2 = locations[-2:]
        if loc1 == "unknown" or loc2 == "unknown":
            return 0.8
            
        l1_info = LOCATIONS[loc1]
        l2_info = LOCATIONS[loc2]
        
        distance = haversine_distance(l1_info["lat"], l1_info["lon"], 
                                    l2_info["lat"], l2_info["lon"])
        
        # 如果距离很远但时间很短，风险很高
        max_reasonable_speed = 800  # 公里/小时（飞机速度）
        if time_diff_hours > 0 and distance / time_diff_hours > max_reasonable_speed:
            return 0.9
        elif distance > 500 and time_diff_hours < 2:
            return 0.7
        elif distance > 100 and time_diff_hours < 0.5:
            return 0.5
        else:
            return 0.1
    
    def generate_complex_transaction(self, txn_id, base_time, account_history=None):
        """生成复杂的交易数据"""
        
        # 选择或创建账户（紧凑模式优先复用账户池）
        if self._compact_mode_enabled:
            # 尽量复用历史账户，否则从账户池中采样
            if account_history and random.random() < max(0.8, self.reuse_prob):
                account_id = account_history["account_id"]
                account = self.accounts[account_id]
            else:
                account_id = random.choice(self.account_pool)
                if account_id not in self.accounts:
                    self.generate_account_profile(account_id)
                account = self.accounts[account_id]
        else:
            if account_history and random.random() < 0.7:
                account_id = account_history["account_id"]
                account = self.accounts[account_id]
            else:
                if self.id_format_account:
                    # 用大随机数保证分散
                    rnd = random.randint(10000, 99999)
                    account_id = self.id_format_account.format(prefix=self.id_prefix_account, type='C', num=rnd)
                else:
                    account_id = f"{self.id_prefix_account}_{random.randint(10000, 99999)}"
                account = self.generate_account_profile(account_id)
            
        # 选择商户类别和商户（紧凑模式从商户池采样）
        if self._compact_mode_enabled and self.merchant_pool:
            merchant_id = random.choice(self.merchant_pool)
            merchant = self.merchants[merchant_id]
            category = merchant["category"]
        else:
            category = random.choices(
                list(MERCHANT_CATEGORIES.keys()), 
                weights=[MERCHANT_CATEGORIES[cat]["weight"] for cat in MERCHANT_CATEGORIES.keys()]
            )[0]
            if self.id_format_merchant:
                rnd = random.randint(1000, 9999)
                merchant_id = self.id_format_merchant.format(prefix=self.id_prefix_merchant, type='S', num=rnd, cat=category)
            else:
                merchant_id = f"{category}_{self.id_prefix_merchant}_{random.randint(1000, 9999)}"
            if merchant_id not in self.merchants:
                merchant = self.generate_merchant_profile(merchant_id, category)
            else:
                merchant = self.merchants[merchant_id]
            
        # 基于商户类别生成金额
        amount_range = MERCHANT_CATEGORIES[category]["avg_amount"]
        base_amount = random.uniform(*amount_range)
        
        # 应用各种风险因素
        risk_multipliers = []
        
        # 1. 账户风险
        if account["is_fraudster"]:
            risk_multipliers.append(("fraudster_account", 2.5))
            base_amount *= random.uniform(0.5, 3.0)  # 欺诈者金额更不规律
            
        # 2. 时间风险
        hour = base_time.hour
        if hour < 6 or hour > 22:
            risk_multipliers.append(("unusual_time", 1.3))
        if base_time.weekday() == 6:  # 周日
            risk_multipliers.append(("weekend", 1.1))
            
        # 3. 地理风险
        if account_history and "last_location" in account_history:
            last_location = account_history["last_location"]
            last_time = account_history["last_time"]
            time_diff = (base_time - last_time).total_seconds() / 3600  # 小时
            
            current_location = random.choice(account["preferred_locations"])
            geo_risk = self.calculate_geographic_risk([last_location, current_location], time_diff)
            if geo_risk > 0.5:
                risk_multipliers.append(("geographic_anomaly", 1.0 + geo_risk))
        else:
            current_location = random.choice(account["preferred_locations"])
            
        # 4. 商户风险
        if merchant["is_suspicious"]:
            risk_multipliers.append(("suspicious_merchant", 1.8))
        if merchant["fraud_history"]:
            risk_multipliers.append(("merchant_fraud_history", 1.5))
            
        # 5. 设备风险
        device_id = random.choice(account["device_ids"])
        if random.random() < 0.1:  # 10%概率使用新设备
            device_id = f"new_device_{uuid.uuid4().hex[:8]}"
            risk_multipliers.append(("new_device", 1.4))
            
        # 6. 频率风险
        if account_history and "recent_transactions" in account_history:
            recent_count = len(account_history["recent_transactions"])
            if recent_count > 10:
                risk_multipliers.append(("high_frequency", 1.2 + recent_count * 0.1))
                
        # 计算最终风险评分
        base_risk = account["risk_baseline"] + merchant["risk_score"]
        for reason, multiplier in risk_multipliers:
            base_risk *= multiplier
            
        # 限制风险评分在0-1范围内
        final_risk_score = min(0.99, max(0.01, base_risk))
        
        # 生成交易数据
        transaction = {
            "id": txn_id,
            "timestamp": base_time.isoformat(),
            "account_id": account_id,
            "merchant_id": merchant_id,
            "merchant_category": category,
            "amount": round(base_amount, 2),
            "currency": random.choices(["CNY", "USD", "EUR"], weights=[0.8, 0.15, 0.05])[0],
            "channel": random.choices(
                ["mobile_app", "web", "pos", "atm", "phone"], 
                weights=[0.4, 0.3, 0.2, 0.08, 0.02]
            )[0],
            "location": current_location,
            "device_id": device_id,
            "risk_factors": dict(risk_multipliers),
            "calculated_risk_score": final_risk_score,
            "is_fraud": final_risk_score > 0.7 or (account["is_fraudster"] and random.random() < 0.3),
            
            # 网络分析相关字段
            "ip_address": f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
            "user_agent": random.choice([
                "Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X)",
                "Mozilla/5.0 (Android 11; Mobile; rv:68.0) Gecko/68.0 Firefox/88.0",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            ]),
            
            # 交易特征
            "mcc_code": f"{random.randint(1000, 9999)}",  # 商户类别码
            "authorization_code": f"AUTH{random.randint(100000, 999999)}",
            "processing_time_ms": random.randint(50, 2000),
            
            # 账户历史特征
            "account_age_days": account["age_days"],
            "account_transaction_count_30d": random.randint(0, 100),
            "account_avg_amount_30d": random.uniform(100, 5000),
            
            # 复杂特征（需要计算的）
            "velocity_features": {
                "transactions_1h": random.randint(0, 20),
                "transactions_24h": random.randint(0, 50),
                "amount_1h": random.uniform(0, 10000),
                "amount_24h": random.uniform(0, 50000)
            }
        }
        
        return transaction, account, current_location
        
    def generate_large_dataset(self, num_transactions=15000):
        """生成大规模复杂数据集"""
        print(f"🚀 开始生成包含{num_transactions}笔交易的复杂数据集...")
        print("📊 这将包含真实的欺诈模式、网络关系和复杂风险因素")
        print("-" * 80)
        
        start_time = time.time()
        transactions = []
        account_histories = {}

        # 若启用紧凑模式，则构建账户/商户池
        self._maybe_build_pools(expected_txn_count=num_transactions)
        
        # 先生成一些欺诈团伙
        print("👥 生成欺诈团伙...")
        for _ in range(random.randint(5, 12)):
            self.generate_fraud_ring()
        
        base_time = datetime.now() - timedelta(days=30)
        
        for i in range(num_transactions):
            if i % 1000 == 0:
                elapsed = time.time() - start_time
                progress = (i / num_transactions) * 100
                print(f"⚡ 进度: {progress:.1f}% ({i}/{num_transactions}) - 耗时: {elapsed:.2f}秒")
                
            # 生成时间戳（模拟30天内的交易）
            txn_time = base_time + timedelta(
                seconds=random.randint(0, 30 * 24 * 60 * 60)
            )
            
            txn_id = f"complex_txn_{i:06d}_{uuid.uuid4().hex[:8]}"
            
            # 70%概率使用已有账户（生成账户历史关联）
            account_history = None
            if account_histories and random.random() < 0.7:
                account_history = random.choice(list(account_histories.values()))
                
            transaction, account, location = self.generate_complex_transaction(
                txn_id, txn_time, account_history
            )
            
            # 更新账户历史
            acc_id = account["account_id"]
            if acc_id not in account_histories:
                account_histories[acc_id] = {
                    "account_id": acc_id,
                    "recent_transactions": [],
                    "last_time": txn_time,
                    "last_location": location
                }
            
            hist = account_histories[acc_id]
            hist["recent_transactions"].append(transaction["id"])
            hist["last_time"] = txn_time
            hist["last_location"] = location
            
            # 只保留最近50笔交易的历史
            if len(hist["recent_transactions"]) > 50:
                hist["recent_transactions"] = hist["recent_transactions"][-50:]
                
            transactions.append(transaction)
            
        total_time = time.time() - start_time
        
        # 生成统计信息
        fraud_count = sum(1 for txn in transactions if txn["is_fraud"])
        high_risk_count = sum(1 for txn in transactions if txn["calculated_risk_score"] > 0.6)
        unique_accounts = len(set(txn["account_id"] for txn in transactions))
        unique_merchants = len(set(txn["merchant_id"] for txn in transactions))
        
        dataset_stats = {
            "generation_time_seconds": total_time,
            "total_transactions": len(transactions),
            "fraud_transactions": fraud_count,
            "fraud_rate": fraud_count / len(transactions),
            "high_risk_transactions": high_risk_count,
            "unique_accounts": unique_accounts,
            "unique_merchants": unique_merchants,
            "fraud_rings": len(self.fraud_rings),
            "avg_transactions_per_account": len(transactions) / unique_accounts,
            "dataset_complexity_score": self.calculate_complexity_score(transactions),
            # 记录紧凑参数与池规模，便于复现
            "compactness": self.compactness,
            "account_pool_size": len(self.account_pool) if self.account_pool else 0,
            "merchant_pool_size": len(self.merchant_pool) if self.merchant_pool else 0,
            "reuse_prob": self.reuse_prob,
        }
        
        print("\n" + "="*80)
        print("📈 数据集生成完成!")
        print(f"⏱️  总耗时: {total_time:.2f}秒")
        print(f"📊 总交易数: {len(transactions):,}")
        print(f"🚨 欺诈交易: {fraud_count} ({fraud_count/len(transactions)*100:.2f}%)")
        print(f"⚠️  高风险交易: {high_risk_count} ({high_risk_count/len(transactions)*100:.2f}%)")
        print(f"👤 独立账户: {unique_accounts:,}")
        print(f"🏪 独立商户: {unique_merchants:,}")
        print(f"👥 欺诈团伙: {len(self.fraud_rings)}")
        print(f"🧮 复杂度评分: {dataset_stats['dataset_complexity_score']:.2f}/10.0")
        print("="*80)
        
        return {
            "transactions": transactions,
            "account_profiles": dict(list(self.accounts.items())[:100]),  # 只保存前100个账户画像
            "merchant_profiles": dict(list(self.merchants.items())[:50]), # 只保存前50个商户画像
            "fraud_rings": self.fraud_rings,
            "statistics": dataset_stats
        }
        
    def calculate_complexity_score(self, transactions):
        """计算数据集复杂度评分"""
        if not transactions:
            return 0.0
            
        # 评估各种复杂度因子
        unique_accounts = len(set(txn["account_id"] for txn in transactions))
        unique_merchants = len(set(txn["merchant_id"] for txn in transactions))
        unique_locations = len(set(txn["location"] for txn in transactions))
        
        fraud_rate = sum(1 for txn in transactions if txn["is_fraud"]) / len(transactions)
        
        # 风险评分分布
        risk_scores = [txn["calculated_risk_score"] for txn in transactions]
        risk_std = (sum((r - sum(risk_scores)/len(risk_scores))**2 for r in risk_scores) / len(risk_scores))**0.5
        
        # 时间分布复杂度
        hours = [datetime.fromisoformat(txn["timestamp"]).hour for txn in transactions]
        hour_distribution = len(set(hours)) / 24.0
        
        # 综合复杂度评分 (0-10)
        complexity_score = (
            min(unique_accounts / 1000, 1.0) * 2.0 +      # 账户多样性
            min(unique_merchants / 500, 1.0) * 1.5 +      # 商户多样性
            min(unique_locations / 10, 1.0) * 1.0 +       # 地理多样性
            min(fraud_rate * 20, 1.0) * 2.0 +             # 欺诈复杂性
            min(risk_std * 5, 1.0) * 2.0 +                # 风险分布复杂性
            hour_distribution * 1.5                        # 时间分布复杂性
        )
        
        return complexity_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成复杂交易数据集")
    parser.add_argument("--count", type=int, default=15000, help="交易条数，默认 15000")
    parser.add_argument("--output", type=str, default="complex_transaction_dataset.json", help="输出文件路径")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可选，指定后可复现）")
    # 紧凑网络相关参数
    parser.add_argument("--compactness", type=float, default=0.0, help="网络紧凑度 0~1，越大图越密集（默认 0）")
    parser.add_argument("--account-pool", dest="account_pool", type=int, default=None, help="唯一账户池大小（默认基于紧凑度推导）")
    parser.add_argument("--merchant-pool", dest="merchant_pool", type=int, default=None, help="唯一商户池大小（默认基于紧凑度推导）")
    parser.add_argument("--reuse-prob", dest="reuse_prob", type=float, default=None, help="复用已有账户/商户的概率 0~1（默认基于紧凑度推导）")
    # 自定义ID命名相关参数
    parser.add_argument("--id-prefix-account", dest="id_prefix_account", type=str, default="acc", help="账户ID前缀（默认 acc）")
    parser.add_argument("--id-prefix-merchant", dest="id_prefix_merchant", type=str, default="mer", help="商户ID前缀（默认 mer）")
    parser.add_argument(
        "--id-format-account", dest="id_format_account", type=str, default=None,
        help="账户ID格式，例如 '{prefix}-{type}-{num:05d}'，可用变量: prefix/type/num"
    )
    parser.add_argument(
        "--id-format-merchant", dest="id_format_merchant", type=str, default=None,
        help="商户ID格式，例如 '{prefix}-{type}-{num:05d}'，可用变量: prefix/type/num/cat"
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    generator = ComplexDatasetGenerator(
        compactness=args.compactness,
        account_pool_size=args.account_pool,
        merchant_pool_size=args.merchant_pool,
        reuse_prob=args.reuse_prob,
        id_prefix_account=args.id_prefix_account,
        id_prefix_merchant=args.id_prefix_merchant,
        id_format_account=args.id_format_account,
        id_format_merchant=args.id_format_merchant,
    )

    # 生成大规模数据集
    dataset = generator.generate_large_dataset(args.count)

    # 保存为JSON文件
    print("💾 保存数据集到文件...")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2, default=str)

    print(f"✅ 数据集已保存为 '{args.output}'")
    print("📁 文件大小约: {:.2f} MB".format(len(json.dumps(dataset, default=str)) / 1024 / 1024))
    print("\n🎯 现在你可以将这个数据集导入系统进行真正的复杂分析了!")