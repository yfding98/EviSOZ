'use strict';

const STATUS_ZH = {
  completed_localizable: '可定位',
  completed_nonlocalizable: '不可定位',
  completed_insufficient_evidence: '证据不足',
  completed_technical_unassessable: '技术不可评价'
};
const CONSISTENCY_ZH = {
  match: '一致', partial_match: '部分一致', mismatch: '不一致', not_available: '不可评价'
};
const LABEL_STATUS_ZH = {
  available: '有标签', not_available: '无可用标签',
  source_conflict: '来源冲突（闭锁）', ambiguous_mapping: '映射歧义（闭锁）'
};
const DISPOSITION_ZH = {
  exact_or_compatible: '一致或相容', mismatch: '不一致',
  generated_abstention: '报告弃权', label_missing: '无标签，无法比较',
  technical_unassessable: '技术不可评价', source_conflict: '来源冲突，不评分',
  ambiguous_mapping: '映射歧义，不评分'
};
const GENERATED_STATUS_ZH = {
  generated_localization: '报告给出定位结论',
  generated_nonfocal_conclusion: '报告给出非局灶性结论',
  generated_abstention: '报告未作定位结论',
  technical_unassessable: '报告技术不可评价'
};
const ALIGNMENT_ZH = {
  exact_spatial_match: '空间定位一致', compatible_spatial_overlap: '空间定位相容',
  spatial_mismatch: '空间定位不一致', uncertainty_aligned: '不确定性一致',
  doctor_uncertain_but_generated_localization: '医生起始不清、报告给出定位',
  generated_abstention_not_scored_as_conflict: '报告弃权，不计作冲突',
  label_not_comparable: '缺少可比较标签', technical_unassessable: '技术不可评价',
  source_conflict_not_scored: '标签来源冲突，不评分'
};
const CODE_ZH = {
  left: '左侧', right: '右侧', bilateral: '双侧', midline: '中线', none: '无侧别', indeterminate: '不确定',
  frontal: '额区', temporal: '颞区', central: '中央区', parietal: '顶区', occipital: '枕区',
  frontotemporal: '额颞区', centrotemporal: '中央颞区', temporoparietal: '颞顶区', posterior: '后部',
  diffuse: '弥散', unknown: '未知', clear: '起始清楚', uncertain_or_unclear: '起始不清/不确定'
};
const RESEARCH_STATUS_ZH = {
  available: '已形成研究候选',
  no_valid_event_rankings: '无有效事件排名',
  technical_unassessable: '技术不可评价',
  not_published: '未附研究候选'
};
const RESEARCH_EVIDENCE_ZH = {
  stable_leading_candidate_descriptive: '未校准稳定首候选',
  limited_cross_event_consistency: '跨事件一致性有限',
  multimodal_or_weak_ranked_hypotheses: '多模式或弱排序证据'
};
const RESEARCH_REASON_ZH = {
  single_mode_repeated_leading_electrode_support: '单一事件模式中首候选重复出现',
  multiple_complete_link_event_modes_detected: '检测到多个事件排序模式',
  fewer_than_three_ranked_events: '可聚合事件少于 3 个',
  leading_electrode_support_below_descriptive_stable_cutpoint: '首候选的跨事件支持不足',
  leading_electrode_top3_support_below_descriptive_stable_cutpoint: '首候选进入事件 Top-3 的支持不足',
  no_event_rankings: '没有可聚合的 EEG 事件排名',
  no_valid_eeg_event_rankings: '没有可聚合的 EEG 事件排名',
  event_research_ranking_validation_failed: '事件排名未通过完整性校验',
  technical_unassessable_bundle_absent: '该记录技术不可评价',
  technical_record_has_no_eeg_candidate_projection: '该记录技术不可评价',
  research_sidecar_not_attached_to_viewer_release: '当前发布包未附研究候选 sidecar'
};

const state = { index: null, filtered: [], selected: null };
const $ = (id) => document.getElementById(id);

function node(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = String(text);
  return el;
}

function badge(value, kind) {
  return node('span', `badge ${kind || badgeKind(value)}`, value);
}

function badgeKind(value) {
  if (value === 'match' || value === 'available' || value === 'completed_localizable') return 'ok';
  if (value === 'mismatch' || value === 'source_conflict') return 'bad';
  if (value === 'partial_match' || value === 'completed_nonlocalizable' || value === 'ambiguous_mapping') return 'warn';
  return 'muted';
}

function statusText(value) { return STATUS_ZH[value] || value; }
function consistencyText(value) { return CONSISTENCY_ZH[value] || value; }
function labelStatusText(value) { return LABEL_STATUS_ZH[value] || value; }
function dispositionText(value) { return DISPOSITION_ZH[value] || value; }
function codes(values) {
  if (!Array.isArray(values) || values.length === 0) return '无可用受控条目';
  return values.map((value) => CODE_ZH[value] || value).join('、');
}

function renderSummary(counts) {
  const specs = [
    ['record_count', '报告总数'], ['subject_count', '被试数'], ['eeg_report_count', '完整 EEG 报告'],
    ['technical_report_count', '技术不可评价'], ['research_soz_candidate_count', '研究性头皮候选'],
    ['physician_label_available_count', '含医生标签'],
    ['physician_label_ambiguous_mapping_count', '标签映射歧义']
  ];
  const root = $('summary');
  root.replaceChildren();
  specs.forEach(([key, label]) => {
    const card = node('article', 'metric');
    card.append(node('strong', '', counts[key]), node('span', '', label));
    root.append(card);
  });
}

function populateStatusFilter(records) {
  const values = [...new Set(records.map((row) => row.diagnostic_status))].sort();
  values.forEach((value) => {
    const option = node('option', '', statusText(value));
    option.value = value;
    $('status-filter').append(option);
  });
}

function applyFilters() {
  const query = $('search').value.trim().toLowerCase();
  const status = $('status-filter').value;
  const label = $('label-filter').value;
  const consistency = $('consistency-filter').value;
  state.filtered = state.index.records.filter((row) => {
    const textMatch = !query || row.recording_id.toLowerCase().includes(query) || row.subject_id.toLowerCase().includes(query);
    return textMatch && (!status || row.diagnostic_status === status)
      && (!label || row.label_status === label)
      && (!consistency || row.location_consistency === consistency);
  });
  renderRows();
}

function renderRows() {
  const body = $('records');
  body.replaceChildren();
  $('result-count').textContent = `显示 ${state.filtered.length} / ${state.index.records.length} 条记录`;
  state.filtered.forEach((row) => {
    const tr = document.createElement('tr');
    tr.tabIndex = 0;
    if (state.selected === row.recording_id) tr.classList.add('active');
    const values = [
      node('span', 'id-cell', row.recording_id),
      node('span', 'id-cell', row.subject_id),
      badge(statusText(row.diagnostic_status), badgeKind(row.diagnostic_status)),
      String(row.event_count),
      row.research_soz_top1
        ? badge(`Top-1 ${row.research_soz_top1}`, 'warn')
        : badge(RESEARCH_STATUS_ZH[row.research_soz_status] || '无候选', 'muted'),
      badge(labelStatusText(row.label_status), badgeKind(row.label_status)),
      badge(consistencyText(row.location_consistency), badgeKind(row.location_consistency))
    ];
    values.forEach((value) => {
      const td = document.createElement('td');
      td.append(value instanceof Node ? value : document.createTextNode(value));
      tr.append(td);
    });
    const open = () => loadDetail(row);
    tr.addEventListener('click', open);
    tr.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); open(); }
    });
    body.append(tr);
  });
}

function fact(label, value) {
  const root = node('div', 'fact');
  root.append(node('span', '', label), node('strong', '', value));
  return root;
}

function chips(values) {
  const root = node('div', 'chips');
  if (!Array.isArray(values) || values.length === 0) root.append(node('span', 'chip', '无可用受控条目'));
  else values.forEach((value) => root.append(node('span', 'chip', CODE_ZH[value] || value)));
  return root;
}

function percent(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '不可计算';
}

function renderResearchCandidates(projection) {
  const panel = section('研究性头皮起始候选（独立于资格化脑电印象）');
  panel.classList.add('research-panel');
  panel.append(node(
    'p',
    'research-boundary',
    '该面板仅汇总 EEG 事件级 C18 电极排序，是未校准的头皮起始假设；不是皮层 SOZ、致痫区或治疗靶点，也不会覆盖下方冻结脑电印象。'
  ));
  const status = projection && projection.status ? projection.status : 'not_published';
  if (status !== 'available') {
    const text = {
      technical_unassessable: '该记录技术不可评价，未形成研究性候选，不输出通道。',
      no_valid_event_rankings: '没有通过校验的有效 EEG 事件排名，未形成研究性候选，不输出通道。',
      not_published: '当前只读发布包未附研究性候选 sidecar；资格化脑电印象仍按原报告展示。'
    }[status] || '未形成研究性候选，不输出通道。';
    panel.append(badge(RESEARCH_STATUS_ZH[status] || status, 'muted'), node('p', 'empty-candidate', text));
    return panel;
  }

  const top = projection.top1_candidate;
  const evidence = RESEARCH_EVIDENCE_ZH[projection.evidence_level] || projection.evidence_level;
  const hero = node('div', 'research-hero');
  const lead = node('div', 'research-lead');
  lead.append(
    node('span', '', '研究性 Top-1'),
    node('strong', '', top.electrode),
    node('small', '', `${top.rank1_support_event_count}/${projection.input_event_count} 个事件列第 1`)
  );
  const evidenceBox = node('div', 'research-evidence');
  evidenceBox.append(node('span', '', '描述性证据等级'), badge(evidence, 'warn'));
  hero.append(lead, evidenceBox);
  panel.append(hero);

  const support = node('div', 'facts');
  support.append(
    fact('有效排序事件', projection.input_event_count),
    fact('Top-1 事件支持', `${top.rank1_support_event_count}/${projection.input_event_count}（${percent(top.rank1_support_rate)}）`),
    fact('进入事件 Top-3', `${top.top3_support_event_count}/${projection.input_event_count}（${percent(top.top3_support_rate)}）`),
    fact('事件模式数', projection.event_support.mode_cluster_count)
  );
  panel.append(support);

  panel.append(node('h4', '', `候选排序 Top-${projection.ranked_candidates.length}`));
  const ranking = node('ol', 'candidate-ranking');
  projection.ranked_candidates.forEach((candidate) => {
    const item = node('li', candidate.rank === 1 ? 'leading' : 'competing');
    item.append(
      node('span', 'rank', `#${candidate.rank}`),
      node('strong', '', candidate.electrode),
      node('span', 'candidate-support', `第1支持 ${candidate.rank1_support_event_count}/${projection.input_event_count} · Top-3支持 ${candidate.top3_support_event_count}/${projection.input_event_count}`)
    );
    ranking.append(item);
  });
  panel.append(ranking);

  if (projection.ranked_candidates.length > 1) {
    panel.append(node('p', 'detail-note', `竞争候选：${projection.ranked_candidates.slice(1).map((item) => item.electrode).join('、')}。`));
  }
  if (projection.event_mode_clusters.length > 1) {
    panel.append(node('h4', '', '不同发作模式'));
    const modes = node('div', 'mode-grid');
    projection.event_mode_clusters.forEach((mode) => {
      modes.append(fact(`模式 ${mode.mode_number} · ${mode.event_count} 个事件`, mode.leading_candidates.join('、')));
    });
    panel.append(modes);
  }
  const reasons = projection.reason_codes.map((code) => RESEARCH_REASON_ZH[code] || code);
  panel.append(node('p', 'detail-note', `证据说明：${reasons.join('；')}。`));
  return panel;
}

function consistencyTable(event) {
  const table = document.createElement('table');
  const head = document.createElement('thead');
  const hr = document.createElement('tr');
  ['字段', '报告', '医生标签', '一致性'].forEach((text) => hr.append(node('th', '', text)));
  head.append(hr); table.append(head);
  const body = document.createElement('tbody');
  const names = { laterality: '侧别', regions: '脑区', onset_uncertainty: '起始清晰度' };
  Object.entries(event.report_fact_consistency).forEach(([field, item]) => {
    const tr = document.createElement('tr');
    [names[field] || field, codes(item.report_values), codes(item.doctor_values)].forEach((text) => tr.append(node('td', '', text)));
    const status = document.createElement('td');
    status.append(badge(consistencyText(item.status), badgeKind(item.status)));
    tr.append(status); body.append(tr);
  });
  table.append(body);
  return table;
}

function renderLabelEvent(event) {
  const card = node('article', 'event-card');
  const title = node('div', 'event-title');
  const identity = event.source_event_slot
    ? `医生标签 ${event.source_event_slot} · 整记录级（不绑定算法候选事件）`
    : `事件 ${event.event_number} · ${event.eeg_event_id}`;
  title.append(node('span', '', identity), badge(event.label_status === 'available' ? '有医生标签' : '无标签', event.label_status === 'available' ? 'ok' : 'muted'));
  const body = node('div', 'event-body');
  const facts = node('div', 'facts');
  facts.append(
    fact('医生起始侧别', codes(event.doctor_onset.laterality)),
    fact('医生起始脑区', codes(event.doctor_onset.regions)),
    fact('医生起始清晰度', codes(event.doctor_onset.onset_uncertainty)),
    fact('定位事实一致性', consistencyText(event.location_consistency)),
    fact('综合事实一致性', consistencyText(event.overall_fact_consistency || event.location_consistency)),
    fact('评价处置', dispositionText(event.evaluation_disposition)),
    fact('报告结论类型', GENERATED_STATUS_ZH[event.generated_report_status] || event.generated_report_status),
    fact('对齐解释', ALIGNMENT_ZH[event.alignment_code] || event.alignment_code)
  );
  body.append(facts);
  const channelTitle = node('p', 'detail-note', '医生通道标签：显著通道为 hard GT；扩散通道仅为 soft label。');
  body.append(channelTitle, node('strong', '', '显著通道'), chips(event.physician_channels.significant), node('strong', '', '扩散通道（soft）'), chips(event.physician_channels.spread_soft_label));
  if (event.physician_channels.diffuse_spread_present) {
    body.append(node('p', 'detail-note', '医生标签另含“弥散/广泛扩散”受控标记；该标记不是起始通道。'));
  }
  body.append(node('p', 'detail-note', '以下只比较冻结报告中的结构化起始结论与医生受控标签；不可用不计为不一致。'), consistencyTable(event));
  card.append(title, body);
  return card;
}

async function loadDetail(row) {
  state.selected = row.recording_id;
  renderRows();
  const root = $('detail');
  root.className = 'detail';
  root.replaceChildren(node('div', 'detail-section', '加载中…'));
  try {
    const response = await fetch(row.detail_url, { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const detail = await response.json();
    renderDetail(detail);
    history.replaceState(null, '', `#${encodeURIComponent(row.recording_id)}`);
  } catch (error) {
    root.replaceChildren(node('div', 'error', `详情加载失败：${error.message}`));
  }
}

function section(title) {
  const root = node('section', 'detail-section');
  root.append(node('h3', '', title));
  return root;
}

function renderDetail(detail) {
  const root = $('detail');
  root.replaceChildren();
  const head = node('div', 'detail-head');
  head.append(node('p', '', detail.subject_id), node('h2', '', detail.recording_id));
  const badges = node('div', 'badge-row');
  badges.append(
    badge(statusText(detail.diagnostic_status), badgeKind(detail.diagnostic_status)),
    badge(`${detail.event_count} 个事件`, 'muted'),
    badge(
      labelStatusText(detail.physician_labels_and_evaluation.status),
      badgeKind(detail.physician_labels_and_evaluation.status)
    )
  );
  head.append(badges); root.append(head);

  root.append(renderResearchCandidates(detail.research_scalp_onset_candidates));

  const labels = section('医生结构化标签与事实一致性');
  labels.append(node('p', 'detail-note', '标签来自报告冻结后的独立评价产物；原始 Excel 自由文本和患者身份未进入此发布包。'));
  const evaluation = detail.physician_labels_and_evaluation;
  const overview = node('div', 'facts');
  overview.append(
    fact('定位一致性', consistencyText(evaluation.location_consistency)),
    fact('起始清晰度一致性', consistencyText(evaluation.onset_certainty_consistency)),
    fact('记录级评价处置', dispositionText(evaluation.record_consistency_disposition)),
    fact('结构化医生标签条目', evaluation.events.filter((event) => event.label_status === 'available').length),
    fact('未匹配参考条目', evaluation.unmatched_reference_count)
  );
  labels.append(overview);
  if (evaluation.status === 'source_conflict') labels.append(node('p', 'detail-note', `该记录的医生标签来源存在冲突，已按策略闭锁 ${evaluation.withheld_conflicting_label_count || 0} 个冲突变体，不展示或用于一致性评价。`));
  else if (evaluation.status === 'ambiguous_mapping') labels.append(node('p', 'detail-note', '同一被试与 SZ 槽位对应多个不同 EEG 信号，无法唯一关联医生标签；已闭锁标签，不展示且不评分。'));
  else if (evaluation.events.length === 0) labels.append(node('p', 'detail-note', '该记录暂无合格的后冻结医生标签评价产物。'));
  else evaluation.events.forEach((event) => labels.append(renderLabelEvent(event)));
  root.append(labels);

  const report = section('冻结诊断报告正文');
  report.append(node('p', 'detail-note', 'AI 草稿，未经脑电医师签署；iframe 已禁用脚本、表单和外部资源。'));
  const frame = document.createElement('iframe');
  frame.className = 'report-frame'; frame.title = `${detail.recording_id} 报告正文`;
  frame.setAttribute('sandbox', 'allow-same-origin');
  frame.loading = 'lazy'; frame.src = detail.report_url;
  report.append(frame); root.append(report);

  if (detail.waveforms.length > 0) {
    const waveformSection = section('EEG 波形证据');
    waveformSection.append(node('p', 'detail-note', '仅展示报告 manifest 哈希校验通过且已移除 PNG 文本元数据的波形；图像仍需医师复核。'));
    const gallery = node('div', 'waveforms');
    detail.waveforms.forEach((waveform, index) => {
      const figure = document.createElement('figure');
      const image = document.createElement('img');
      image.src = waveform.url; image.loading = 'lazy'; image.alt = `事件 ${index + 1} EEG 波形`;
      figure.append(image, node('figcaption', '', waveform.name)); gallery.append(figure);
    });
    waveformSection.append(gallery); root.append(waveformSection);
  }
}

async function boot() {
  try {
    const response = await fetch('data/index.json', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.index = await response.json();
    renderSummary(state.index.counts);
    populateStatusFilter(state.index.records);
    ['search', 'status-filter', 'label-filter', 'consistency-filter'].forEach((id) => {
      $(id).addEventListener(id === 'search' ? 'input' : 'change', applyFilters);
    });
    applyFilters();
    const requested = decodeURIComponent(location.hash.slice(1));
    const row = state.index.records.find((item) => item.recording_id === requested);
    if (row) loadDetail(row);
  } catch (error) {
    $('records').replaceChildren(node('tr', '', `发布包索引加载失败：${error.message}`));
    $('result-count').textContent = '加载失败';
  }
}

boot();
