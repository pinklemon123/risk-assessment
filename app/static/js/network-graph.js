// 极简黑白D3交易网络可视化器（力导向图）
class NetworkGraphVisualizer {
  constructor(containerId, width = 800, height = 400) {
    this.containerId = containerId;
    this.width = width;
    this.height = height;
    this.nodes = [];
    this.links = [];
    this.colors = { high: '#000', medium: '#666', low: '#ccc', link: '#999' };
  // 标签显示控制：默认仅显示高风险节点标签
  this.showAllLabels = false;
  // 形状与描边控制
  this.shapeByType = false; // 账户:圆, 商户:方
  this.boldHighRisk = false; // 高风险加粗描边

    const container = document.getElementById(containerId);
    container.innerHTML = '';

    // SVG + zoom
    this.svg = d3.select(container)
      .append('svg')
      .attr('width', '100%')
      .attr('height', '100%')
      .attr('viewBox', `0 0 ${width} ${height}`)
      .style('border', '1px solid #ddd')
      .style('background', '#fff');

    this.mainGroup = this.svg.append('g');
    const zoom = d3.zoom().scaleExtent([0.2, 5]).on('zoom', (ev) => {
      this.mainGroup.attr('transform', ev.transform);
    });
    this.svg.call(zoom);

    // Simulation
    this.simulation = d3.forceSimulation()
      .force('link', d3.forceLink().id(d => d.id).distance(80))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(22));
  }

  renderNetwork() {
    if (!this.nodes || !this.links || this.nodes.length === 0) {
      this.showNoDataMessage();
      return;
    }
    this.simulation.nodes(this.nodes);
    this.simulation.force('link').links(this.links);
    this.updateElements(this.nodes, this.links);
    this.simulation.alpha(1).restart();
  }

  updateElements(nodes, links) {
    this.mainGroup.selectAll('*').remove();

    this.linkElements = this.mainGroup.append('g')
      .selectAll('line')
      .data(links)
      .enter().append('line')
      .attr('stroke', this.colors.link)
      .attr('stroke-width', d => Math.max(1, Math.sqrt(d.value || 1)))
      .attr('opacity', 0.6);

    const self = this;
    // 节点形状：根据 shapeByType 决定使用 circle 或 rect
    const nodeGroup = this.mainGroup.append('g');
    if (this.shapeByType) {
      // account -> circle
      const accounts = nodeGroup.selectAll('circle')
        .data(nodes.filter(n => n.group === 'account'))
        .enter().append('circle')
        .attr('r', d => this.getNodeRadius(d))
        .attr('fill', d => this.getNodeColor(d.riskLevel))
        .attr('stroke', '#000')
        .attr('stroke-width', d => this.getStrokeWidth(d))
        .call(this.createDragBehavior())
        .on('click', (ev, d) => { if (typeof window.onGraphNodeClick === 'function') window.onGraphNodeClick(d); });

      // merchant -> rect（用宽高近似圆大小）
      const merchants = nodeGroup.selectAll('rect')
        .data(nodes.filter(n => n.group === 'merchant'))
        .enter().append('rect')
        .attr('width', d => this.getNodeRadius(d) * 2)
        .attr('height', d => this.getNodeRadius(d) * 2)
        .attr('x', d => (d.x || 0) - this.getNodeRadius(d))
        .attr('y', d => (d.y || 0) - this.getNodeRadius(d))
        .attr('rx', 2).attr('ry', 2)
        .attr('fill', d => this.getNodeColor(d.riskLevel))
        .attr('stroke', '#000')
        .attr('stroke-width', d => this.getStrokeWidth(d))
        .call(this.createDragBehavior())
        .on('click', (ev, d) => { if (typeof window.onGraphNodeClick === 'function') window.onGraphNodeClick(d); });

      // 合并 selection 以便 tick 时统一更新
      this.nodeElements = accounts.merge(merchants);
    } else {
      this.nodeElements = nodeGroup
        .selectAll('circle')
        .data(nodes)
        .enter().append('circle')
        .attr('r', d => this.getNodeRadius(d))
        .attr('fill', d => this.getNodeColor(d.riskLevel))
        .attr('stroke', '#000')
        .attr('stroke-width', d => this.getStrokeWidth(d))
        .call(this.createDragBehavior())
        .on('click', (ev, d) => { if (typeof window.onGraphNodeClick === 'function') window.onGraphNodeClick(d); });
    }

    // 悬停提示：显示完整ID/类型/风险/度数
    this.nodeElements.append('title')
      .text(d => `${d.id} [${d.group}]\n风险: ${d.riskLevel}  连接度: ${d.degree}`);

    // 标签：根据 showAllLabels 决定显示所有或仅高风险节点
    const formatLabel = (d) => {
      const id = d.id || '';
      // 若ID结尾包含较长数字，优先显示末尾数字
      const m = id.match(/(\d{4,})$/);
      if (m) return m[1];
      // 否则去除常见前缀后，显示更长的截断（最多12字符）
      return id.replace(/^(acc_|merchant_|pool_)/, '').slice(0, 12);
    };

    const labelNodes = this.showAllLabels ? nodes : nodes.filter(d => d.riskLevel === 'high');
    this.textElements = this.mainGroup.append('g')
      .selectAll('text')
      .data(labelNodes)
      .enter().append('text')
      .text(d => formatLabel(d))
      .attr('font-size', '10px')
      .attr('text-anchor', 'middle')
      .attr('fill', '#000');

    this.simulation.on('tick', () => {
      this.linkElements
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      if (this.shapeByType) {
        // circle nodes（accounts）
        this.mainGroup.selectAll('circle')
          .attr('cx', d => d.x)
          .attr('cy', d => d.y);
        // rect nodes（merchants）
        this.mainGroup.selectAll('rect')
          .attr('x', d => d.x - this.getNodeRadius(d))
          .attr('y', d => d.y - this.getNodeRadius(d));
      } else {
        this.nodeElements
          .attr('cx', d => d.x)
          .attr('cy', d => d.y);
      }

      this.textElements
        .attr('x', d => d.x)
        .attr('y', d => d.y - 18);
    });
  }

  getNodeRadius(node) {
    const base = node.group === 'account' ? 8 : 6;
    const degree = Math.max(1, Math.min(2, (node.degree || 1) / 10));
    return base * degree;
  }

  getNodeColor(level) {
    if (level === 'high') return this.colors.high;
    if (level === 'medium') return this.colors.medium;
    return this.colors.low;
  }

  createDragBehavior() {
    const sim = this.simulation;
    return d3.drag()
      .on('start', (ev, d) => { if (!ev.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on('drag', (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
      .on('end', (ev, d) => { if (!ev.active) sim.alphaTarget(0); d.fx = null; d.fy = null; });
  }

  getStrokeWidth(node) {
    if (this.boldHighRisk && node.riskLevel === 'high') return 2.5;
    return 1;
  }

  showNoDataMessage() {
    this.mainGroup.selectAll('*').remove();
    this.mainGroup.append('text')
      .attr('x', this.width / 2)
      .attr('y', this.height / 2)
      .attr('text-anchor', 'middle')
      .style('font-size', '14px')
      .style('fill', '#666')
      .text('暂无网络数据');
  }

  setShowAllLabels(flag) {
    this.showAllLabels = !!flag;
    // 重新渲染元素以应用标签显示规则
    this.renderNetwork();
  }

  setShapeByType(flag) {
    this.shapeByType = !!flag;
    this.renderNetwork();
  }

  setBoldHighRisk(flag) {
    this.boldHighRisk = !!flag;
    this.renderNetwork();
  }
}