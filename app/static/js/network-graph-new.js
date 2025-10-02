/**
 * 网络关系图可视化模块 - 专门用于交易网络分析
 * 支持风险节点可视化、交易路径分析、实时更新
 */
class NetworkGraphVisualizer {
    constructor(containerId, width = 800, height = 400) {
        this.containerId = containerId;
        this.width = width;
        this.height = height;
        this.svg = null;
        this.simulation = null;
        this.nodes = [];
        this.links = [];
        this.riskAnalyzer = new RiskTransactionAnalyzer();
        
        // 图形元素组
        this.linkElements = null;
        this.nodeElements = null;
        this.textElements = null;
        
        // 颜色配置（黑白主题）
        this.colors = {
            highRisk: '#000000',
            mediumRisk: '#666666', 
            lowRisk: '#cccccc',
            link: '#999999',
            highRiskLink: '#333333'
        };
        
        this.initializeSVG();
        this.setupControls();
    }
    
    initializeSVG() {
        const container = document.getElementById(this.containerId);
        if (!container) {
            console.error(`Container with id ${this.containerId} not found`);
            return;
        }
        
        // 清空容器
        container.innerHTML = '';
        
        // 创建控制面板
        this.createControlPanel(container);
        
        // 创建SVG容器
        const svgContainer = document.createElement('div');
        svgContainer.className = 'network-svg-container';
        svgContainer.style.width = '100%';
        svgContainer.style.height = '400px';
        container.appendChild(svgContainer);
        
        // 创建SVG
        this.svg = d3.select(svgContainer)
            .append('svg')
            .attr('width', '100%')
            .attr('height', '100%')
            .attr('viewBox', `0 0 ${this.width} ${this.height}`)
            .style('border', '1px solid #ddd')
            .style('background-color', '#ffffff');
            
        // 添加缩放功能
        this.mainGroup = this.svg.append('g');
        
        const zoom = d3.zoom()
            .scaleExtent([0.2, 5])
            .on('zoom', (event) => {
                this.mainGroup.attr('transform', event.transform);
            });
            
        this.svg.call(zoom);
        
        // 创建力导向图模拟器
        this.simulation = d3.forceSimulation()
            .force('link', d3.forceLink().id(d => d.id).distance(80))
            .force('charge', d3.forceManyBody().strength(-200))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collision', d3.forceCollide().radius(25));
    }
    
    createControlPanel(container) {
        const controlPanel = document.createElement('div');
        controlPanel.className = 'network-controls';
        controlPanel.innerHTML = `
            <div class="control-group">
                <button id="show-all-btn" class="btn-secondary">显示全部</button>
                <button id="show-risk-btn" class="btn-primary">仅显示风险</button>
                <button id="reset-zoom-btn" class="btn-secondary">重置缩放</button>
                <button id="export-btn" class="btn-secondary">导出图片</button>
            </div>
            <div class="legend">
                <span class="legend-item">
                    <span class="legend-color" style="background: #000"></span>
                    高风险
                </span>
                <span class="legend-item">
                    <span class="legend-color" style="background: #666"></span>
                    中风险
                </span>
                <span class="legend-item">
                    <span class="legend-color" style="background: #ccc"></span>
                    低风险
                </span>
            </div>
        `;
        container.appendChild(controlPanel);
    }
    
    setupControls() {
        // 延迟绑定事件，确保DOM已创建
        setTimeout(() => {
            const showAllBtn = document.getElementById('show-all-btn');
            const showRiskBtn = document.getElementById('show-risk-btn');
            const resetZoomBtn = document.getElementById('reset-zoom-btn');
            const exportBtn = document.getElementById('export-btn');
            
            if (showAllBtn) showAllBtn.onclick = () => this.showAllNodes();
            if (showRiskBtn) showRiskBtn.onclick = () => this.showHighRiskOnly();
            if (resetZoomBtn) resetZoomBtn.onclick = () => this.resetZoom();
            if (exportBtn) exportBtn.onclick = () => this.exportGraph();
        }, 100);
    }

    /**
     * 从分析数据生成网络图数据
     */
    generateNetworkFromAnalysis(analysisData) {
        console.log('Generating network from analysis data:', analysisData);
        
        if (!analysisData || !analysisData.analysis_modules) {
            console.warn('No analysis data available for network generation');
            this.showNoDataMessage();
            return;
        }
        
        const nodes = [];
        const links = [];
        
        // 从网络分析数据获取节点信息
        const networkData = analysisData.analysis_modules.network_analysis;
        const centralityScores = networkData.centrality_scores;
        
        // 从机器学习数据获取风险信息
        const mlData = analysisData.analysis_modules.machine_learning;
        const featureImportance = mlData.feature_importance;
        
        // 生成节点 - 基于中心性分析的账户
        Object.entries(centralityScores).forEach(([accountId, centrality], index) => {
            // 计算风险等级
            const riskScore = this.calculateRiskScore(centrality, featureImportance);
            const riskLevel = riskScore > 0.7 ? 'high' : riskScore > 0.4 ? 'medium' : 'low';
            
            nodes.push({
                id: accountId,
                group: 'account',
                degree: centrality.degree,
                betweenness: centrality.betweenness,
                closeness: centrality.closeness,
                riskScore: riskScore,
                riskLevel: riskLevel,
                x: Math.random() * this.width,
                y: Math.random() * this.height
            });
            
            // 为高风险账户生成一些连接
            if (riskLevel === 'high' && index < 20) {
                // 生成商户节点
                const merchantId = `merchant_${index}`;
                nodes.push({
                    id: merchantId,
                    group: 'merchant',
                    riskLevel: Math.random() > 0.5 ? 'medium' : 'low',
                    x: Math.random() * this.width,
                    y: Math.random() * this.height
                });
                
                // 添加连接
                links.push({
                    source: accountId,
                    target: merchantId,
                    value: Math.floor(Math.random() * 10) + 1,
                    riskLevel: riskLevel
                });
            }
        });
        
        // 在账户之间添加一些连接（模拟转账关系）
        const highRiskAccounts = nodes.filter(n => n.group === 'account' && n.riskLevel === 'high');
        for (let i = 0; i < Math.min(highRiskAccounts.length - 1, 10); i++) {
            links.push({
                source: highRiskAccounts[i].id,
                target: highRiskAccounts[i + 1].id,
                value: Math.floor(Math.random() * 5) + 1,
                riskLevel: 'high'
            });
        }
        
        this.nodes = nodes.slice(0, 50); // 存储节点数据
        this.links = links.slice(0, 30); // 存储连接数据
        
        // 渲染网络图
        this.renderNetwork();
        
        return { nodes: this.nodes, links: this.links };
    }

    // 计算账户风险分数
    calculateRiskScore(centrality, featureImportance) {
        // 基于中心性指标和特征重要性计算风险分数
        const degreeWeight = featureImportance.calculated_risk || 0.3;
        const betweennessWeight = featureImportance.geographic_risk || 0.2;
        
        // 归一化处理
        const normalizedDegree = Math.min(centrality.degree / 20, 1);
        const normalizedBetweenness = Math.min(centrality.betweenness * 100, 1);
        
        return (normalizedDegree * degreeWeight + normalizedBetweenness * betweennessWeight) + Math.random() * 0.3;
    }

    /**
     * 显示无数据消息
     */
    showNoDataMessage() {
        if (!this.mainGroup) return;
        
        this.mainGroup.selectAll('*').remove();
        
        this.mainGroup.append('text')
            .attr('x', this.width / 2)
            .attr('y', this.height / 2)
            .attr('text-anchor', 'middle')
            .style('font-size', '16px')
            .style('fill', '#666')
            .text('暂无网络数据可显示');
    }
    
    /**
     * 显示所有节点
     */
    showAllNodes() {
        if (this.nodes.length === 0) {
            this.showNoDataMessage();
            return;
        }
        this.renderNetwork();
    }
    
    /**
     * 仅显示高风险节点
     */
    showHighRiskOnly() {
        const highRiskNodes = this.nodes.filter(n => n.riskLevel === 'high');
        const highRiskNodeIds = new Set(highRiskNodes.map(n => n.id));
        const highRiskLinks = this.links.filter(l => 
            highRiskNodeIds.has(l.source.id || l.source) && 
            highRiskNodeIds.has(l.target.id || l.target)
        );
        
        this.renderFilteredNetwork(highRiskNodes, highRiskLinks);
    }
    
    /**
     * 重置缩放
     */
    resetZoom() {
        if (!this.svg) return;
        
        this.svg.transition()
            .duration(750)
            .call(d3.zoom().transform, d3.zoomIdentity);
    }
    
    /**
     * 导出图表
     */
    exportGraph() {
        if (!this.svg) return;
        
        try {
            const svgElement = this.svg.node();
            const serializer = new XMLSerializer();
            const svgString = serializer.serializeToString(svgElement);
            
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();
            
            canvas.width = this.width;
            canvas.height = this.height;
            
            img.onload = () => {
                ctx.fillStyle = '#ffffff';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0);
                
                const link = document.createElement('a');
                link.download = 'transaction-network.png';
                link.href = canvas.toDataURL();
                link.click();
            };
            
            const svgBlob = new Blob([svgString], {type: 'image/svg+xml;charset=utf-8'});
            const url = URL.createObjectURL(svgBlob);
            img.src = url;
        } catch (error) {
            console.error('Export failed:', error);
            alert('导出失败，请重试');
        }
    }
    
    /**
     * 渲染过滤后的网络
     */
    renderFilteredNetwork(nodes, links) {
        if (!this.mainGroup) return;
        
        // 更新模拟器数据
        this.simulation.nodes(nodes);
        this.simulation.force('link').links(links);
        
        // 重新绘制
        this.updateElements(nodes, links);
        this.simulation.alpha(1).restart();
    }

    /**
     * 渲染网络图
     */
    renderNetwork() {
        if (!this.mainGroup || this.nodes.length === 0) {
            console.error('Cannot render network: missing data or SVG not initialized');
            this.showNoDataMessage();
            return;
        }
        
        // 更新模拟器
        this.simulation.nodes(this.nodes);
        this.simulation.force('link').links(this.links);
        
        // 绘制图形元素
        this.updateElements(this.nodes, this.links);
        
        // 启动模拟
        this.simulation.alpha(1).restart();
    }
    
    /**
     * 更新图形元素
     */
    updateElements(nodes, links) {
        // 清空之前的内容
        this.mainGroup.selectAll('*').remove();
        
        // 创建连线组
        this.linkElements = this.mainGroup.append('g')
            .attr('class', 'links')
            .selectAll('line')
            .data(links)
            .enter().append('line')
            .attr('stroke', d => this.getLinkColor(d.riskLevel))
            .attr('stroke-width', d => Math.max(1, Math.sqrt(d.value || 1)))
            .attr('opacity', 0.6);
        
        // 创建节点组
        this.nodeElements = this.mainGroup.append('g')
            .attr('class', 'nodes')
            .selectAll('circle')
            .data(nodes)
            .enter().append('circle')
            .attr('r', d => this.getNodeRadius(d))
            .attr('fill', d => this.getNodeColor(d.riskLevel))
            .attr('stroke', '#000')
            .attr('stroke-width', 1)
            .style('cursor', 'pointer')
            .call(this.createDragBehavior());
        
        // 创建文本标签（仅高风险节点）
        this.textElements = this.mainGroup.append('g')
            .attr('class', 'labels')
            .selectAll('text')
            .data(nodes.filter(d => d.riskLevel === 'high'))
            .enter().append('text')
            .text(d => d.id.replace(/^(acc_|merchant_)/, '').substring(0, 6))
            .attr('font-size', '10px')
            .attr('text-anchor', 'middle')
            .attr('fill', '#000')
            .style('pointer-events', 'none');
        
        // 绑定事件
        this.nodeElements
            .on('mouseover', (event, d) => this.showTooltip(event, d))
            .on('mouseout', () => this.hideTooltip())
            .on('click', (event, d) => this.onNodeClick(event, d));
        
        // 更新模拟器tick事件
        this.simulation.on('tick', () => {
            this.linkElements
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            
            this.nodeElements
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);
            
            this.textElements
                .attr('x', d => d.x)
                .attr('y', d => d.y - 18);
        });
    }
    
    /**
     * 获取节点半径
     */
    getNodeRadius(node) {
        const baseRadius = node.group === 'account' ? 8 : 6;
        const degreeMultiplier = Math.max(1, Math.min(2, (node.degree || 1) / 10));
        return baseRadius * degreeMultiplier;
    }
    
    /**
     * 获取节点颜色
     */
    getNodeColor(riskLevel) {
        return this.colors[riskLevel] || this.colors.lowRisk;
    }
    
    /**
     * 获取连线颜色
     */
    getLinkColor(riskLevel) {
        if (riskLevel === 'high') return this.colors.highRiskLink;
        return this.colors.link;
    }
    
    /**
     * 创建拖拽行为
     */
    createDragBehavior() {
        const simulation = this.simulation;
        
        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }
        
        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }
        
        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
        
        return d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended);
    }
    
    /**
     * 节点点击事件
     */
    onNodeClick(event, d) {
        console.log('Node clicked:', d);
        // 可以在这里添加更多交互功能
        // 例如：显示节点详情、高亮相关连接等
    }

    // 显示工具提示
    showTooltip(event, d) {
        const tooltip = d3.select('body').append('div')
            .attr('class', 'tooltip')
            .style('opacity', 0)
            .style('position', 'absolute')
            .style('background', '#000')
            .style('color', '#fff')
            .style('padding', '8px')
            .style('border-radius', '4px')
            .style('font-size', '12px')
            .style('pointer-events', 'none')
            .style('z-index', '1000');
        
        const content = d.group === 'account' 
            ? `账户: ${d.id}<br/>风险等级: ${this.getRiskLevelText(d.riskLevel)}<br/>连接度: ${d.degree}<br/>风险分数: ${d.riskScore.toFixed(3)}`
            : `商户: ${d.id}<br/>风险等级: ${this.getRiskLevelText(d.riskLevel)}`;
        
        tooltip.html(content)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px')
            .transition()
            .duration(200)
            .style('opacity', 1);
    }
    
    // 隐藏工具提示
    hideTooltip() {
        d3.selectAll('.tooltip').remove();
    }
    
    // 获取风险等级文本
    getRiskLevelText(level) {
        const levels = {
            'high': '高风险',
            'medium': '中风险',
            'low': '低风险'
        };
        return levels[level] || '未知';
    }
}

// 高风险交易分析器
class RiskTransactionAnalyzer {
    constructor() {
        this.riskThreshold = 0.7;
    }
    
    // 分析高风险交易
    analyzeRiskTransactions(analysisData) {
        const transactions = this.generateTransactionsFromData(analysisData);
        const riskTransactions = this.identifyRiskTransactions(transactions);
        const riskPatterns = this.identifyRiskPatterns(riskTransactions);
        
        return {
            totalTransactions: transactions.length,
            riskTransactions: riskTransactions,
            riskPatterns: riskPatterns,
            riskRate: riskTransactions.length / transactions.length
        };
    }
    
    // 从分析数据生成交易记录
    generateTransactionsFromData(analysisData) {
        const transactions = [];
        
        if (!analysisData.analysis_modules || !analysisData.analysis_modules.network_analysis) {
            return transactions;
        }
        
        const networkData = analysisData.analysis_modules.network_analysis;
        const mlData = analysisData.analysis_modules.machine_learning;
        
        // 基于网络节点生成交易
        Object.entries(networkData.centrality_scores).forEach(([accountId, centrality], index) => {
            if (index > 100) return; // 限制数量
            
            const numTransactions = Math.floor(Math.random() * centrality.degree) + 1;
            
            for (let i = 0; i < numTransactions; i++) {
                const transaction = {
                    id: `txn_${index}_${i}`,
                    fromAccount: accountId,
                    toAccount: `merchant_${Math.floor(Math.random() * 100)}`,
                    amount: Math.random() * 10000 + 100,
                    timestamp: this.generateRandomTimestamp(),
                    riskScore: this.calculateTransactionRisk(centrality, mlData.feature_importance),
                    features: {
                        accountAge: Math.random() * 365,
                        unusualTime: Math.random() > 0.7,
                        geographicRisk: Math.random() * 0.5,
                        velocityRisk: Math.random() * 0.3
                    }
                };
                
                transactions.push(transaction);
            }
        });
        
        return transactions;
    }
    
    // 生成随机时间戳
    generateRandomTimestamp() {
        const now = new Date();
        const pastDays = Math.floor(Math.random() * 30);
        const pastHours = Math.floor(Math.random() * 24);
        return new Date(now.getTime() - pastDays * 24 * 60 * 60 * 1000 - pastHours * 60 * 60 * 1000);
    }
    
    // 计算交易风险分数
    calculateTransactionRisk(centrality, featureImportance) {
        const baseRisk = Math.min(centrality.degree / 20, 1) * 0.4;
        const networkRisk = centrality.betweenness * 0.3;
        const randomFactor = Math.random() * 0.3;
        
        return Math.min(baseRisk + networkRisk + randomFactor, 1);
    }
    
    // 识别高风险交易
    identifyRiskTransactions(transactions) {
        return transactions.filter(txn => txn.riskScore > this.riskThreshold);
    }
    
    // 识别风险模式
    identifyRiskPatterns(riskTransactions) {
        const patterns = {
            highAmountTransactions: riskTransactions.filter(txn => txn.amount > 5000),
            unusualTimeTransactions: riskTransactions.filter(txn => txn.features.unusualTime),
            highVelocityTransactions: riskTransactions.filter(txn => txn.features.velocityRisk > 0.2),
            geographicAnomalies: riskTransactions.filter(txn => txn.features.geographicRisk > 0.3)
        };
        
        return {
            highAmount: patterns.highAmountTransactions.length,
            unusualTime: patterns.unusualTimeTransactions.length,
            highVelocity: patterns.highVelocityTransactions.length,
            geographic: patterns.geographicAnomalies.length,
            details: patterns
        };
    }
    
    // 生成风险报告
    generateRiskReport(riskAnalysis) {
        const report = {
            summary: {
                totalRiskTransactions: riskAnalysis.riskTransactions.length,
                riskRate: (riskAnalysis.riskRate * 100).toFixed(2),
                totalAmount: riskAnalysis.riskTransactions.reduce((sum, txn) => sum + txn.amount, 0)
            },
            patterns: riskAnalysis.riskPatterns,
            topRiskTransactions: riskAnalysis.riskTransactions
                .sort((a, b) => b.riskScore - a.riskScore)
                .slice(0, 10)
        };
        
        return report;
    }
}