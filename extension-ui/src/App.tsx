import { useState, useEffect, useRef, useCallback } from 'react'
import * as d3 from 'd3'
import {
  AnalysisState, DEFAULT_STATE, MOCK_STATE, IS_EXTENSION,
  getCurrentTabId, getTabState, triggerAnalysis,
  askChat, onStateUpdate, loadConfig,
} from './lib/chrome'

type View = 'chat' | 'graph'

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  text: string
  typing?: boolean
  followups?: string[]
  citations?: Array<{ title: string; url: string }>
}

// ── Markdown renderer ─────────────────────────────────────────────────────────
function inlineMarkdown(text: string): React.ReactNode {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/)
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**'))
      return <strong key={i} style={{ color: '#e4e4e7', fontWeight: 600 }}>{part.slice(2, -2)}</strong>
    if (part.startsWith('`') && part.endsWith('`'))
      return <code key={i} style={{ background: '#1a1a2e', padding: '1px 5px', borderRadius: 4, fontSize: '0.9em', fontFamily: 'ui-monospace,monospace', color: '#93c5fd' }}>{part.slice(1, -1)}</code>
    return part
  })
}

function renderMarkdown(text: string): React.ReactNode {
  const lines = text.split('\n')
  const nodes: React.ReactNode[] = []
  let i = 0
  while (i < lines.length) {
    const line = lines[i]
    if (/^[-•*]\s/.test(line)) {
      const items: string[] = []
      while (i < lines.length && /^[-•*]\s/.test(lines[i]))
        items.push(lines[i++].replace(/^[-•*]\s/, ''))
      nodes.push(
        <ul key={`ul${i}`} style={{ margin: '4px 0 6px', paddingLeft: 16 }}>
          {items.map((it, j) => <li key={j} style={{ marginBottom: 2 }}>{inlineMarkdown(it)}</li>)}
        </ul>
      )
      continue
    }
    if (/^\d+\.\s/.test(line)) {
      const items: string[] = []
      while (i < lines.length && /^\d+\.\s/.test(lines[i]))
        items.push(lines[i++].replace(/^\d+\.\s/, ''))
      nodes.push(
        <ol key={`ol${i}`} style={{ margin: '4px 0 6px', paddingLeft: 16 }}>
          {items.map((it, j) => <li key={j} style={{ marginBottom: 2 }}>{inlineMarkdown(it)}</li>)}
        </ol>
      )
      continue
    }
    if (line.trim() === '') {
      nodes.push(<span key={`sp${i}`} style={{ display: 'block', height: 6 }} />)
      i++
      continue
    }
    nodes.push(<span key={`ln${i}`} style={{ display: 'block' }}>{inlineMarkdown(line)}</span>)
    i++
  }
  return <>{nodes}</>
}

// ── Typewriter ────────────────────────────────────────────────────────────────
function TypewriterText({ text, speed = 8, onDone }: { text: string; speed?: number; onDone?: () => void }) {
  const [shown, setShown] = useState('')
  const doneRef = useRef(false)
  useEffect(() => {
    doneRef.current = false
    setShown('')
    let i = 0
    const iv = setInterval(() => {
      i += 3
      setShown(text.slice(0, i))
      if (i >= text.length) {
        clearInterval(iv)
        if (!doneRef.current) { doneRef.current = true; onDone?.() }
      }
    }, speed)
    return () => clearInterval(iv)
  }, [text])
  return <span>{shown}<span style={{ opacity: shown.length < text.length ? 1 : 0, borderRight: '2px solid #3b82f6', marginLeft: 1 }} /></span>
}

// ── Copy Button ───────────────────────────────────────────────────────────────
function CopyBtn({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  const copy = () => {
    navigator.clipboard.writeText(text)
      .then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500) })
      .catch(() => {})
  }
  return (
    <button onClick={copy} title="Copy response"
      style={{ background: 'transparent', border: 'none', padding: '2px 6px', cursor: 'pointer', color: copied ? '#22c55e' : '#3f3f46', fontSize: 10, fontFamily: 'inherit', borderRadius: 5, transition: 'color 0.15s', display: 'flex', alignItems: 'center', gap: 3 }}
      onMouseEnter={e => { if (!copied) e.currentTarget.style.color = '#52525b' }}
      onMouseLeave={e => { if (!copied) e.currentTarget.style.color = '#3f3f46' }}>
      {copied
        ? <svg width="10" height="10" viewBox="0 0 10 10" fill="none"><path d="M1.5 5l2.5 2.5 4.5-4.5" stroke="#22c55e" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
        : <svg width="10" height="10" viewBox="0 0 10 10" fill="none"><rect x="1" y="3" width="6" height="6" rx="1" stroke="currentColor" strokeWidth="1.1" /><path d="M3 3V2a1 1 0 011-1h4a1 1 0 011 1v5a1 1 0 01-1 1H7" stroke="currentColor" strokeWidth="1.1" /></svg>}
      {copied ? 'Copied' : 'Copy'}
    </button>
  )
}

// ── Status Strip ──────────────────────────────────────────────────────────────
function StatusStrip({ state }: { state: AnalysisState }) {
  if (state.status !== 'complete' && state.status !== 'partial') return null
  const pills: Array<{ label: string; value: string; color?: string }> = []
  if (typeof state.confidence === 'number') {
    const pct = Math.round(state.confidence * 100)
    pills.push({ label: 'confidence', value: `${pct}%`, color: pct >= 70 ? '#22c55e' : pct >= 40 ? '#f59e0b' : '#ef4444' })
  }
  if (state.current_paper_indexed)
    pills.push({ label: 'indexed', value: `${state.current_paper_chunks ?? 0} chunks`, color: '#22c55e' })
  else if (state.current_paper_indexing)
    pills.push({ label: 'indexing', value: '…', color: '#f59e0b' })
  if (state.from_cache) pills.push({ label: 'cached', value: '●', color: '#3b82f6' })
  if (state.provider) pills.push({ label: 'via', value: state.provider })
  if (pills.length === 0) return null
  return (
    <div style={{ display: 'flex', gap: 5, padding: '0 14px 8px', flexWrap: 'wrap' }}>
      {pills.map((p, i) => (
        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 4, padding: '2px 8px', borderRadius: 99, background: '#111113', border: '1px solid #1f1f23' }}>
          <span style={{ fontSize: 9, color: '#3f3f46', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{p.label}</span>
          <span style={{ fontSize: 10, color: p.color ?? '#52525b', fontWeight: 600 }}>{p.value}</span>
        </div>
      ))}
    </div>
  )
}

function notifySidebarReady() {
  if (!IS_EXTENSION) return
  try { (window as any).chrome.runtime.sendMessage({ type: 'SIDEBAR_READY' }).catch(() => {}) } catch {}
}

let _msgId = 0
const nextId = () => String(++_msgId)

// ── D3 Mind Map ───────────────────────────────────────────────────────────────
interface MindNode {
  id: string
  label: string
  type: 'root' | 'category' | 'leaf'
  category?: string
  url?: string
  children?: MindNode[]
}

const CATEGORY_COLORS: Record<string, string> = {
  domain:  '#3b82f6',
  concept: '#a855f7',
  paper:   '#06b6d4',
  root:    '#e4e4e7',
}

const BAD_LABEL = (t: string) => !t || t.trim().length < 4 || t === 'Untitled'
const JUNK_WORDS = new Set([
  'the','and','for','with','from','that','this','using','based',
  'model','paper','method','approach','work','show','novel','new',
  'results','need','study','learning','network','data','system',
])

function buildMindTree(state: AnalysisState): MindNode {
  const paperTitle = (state.paper_title || '').trim()
  const paperTitleLower = paperTitle.toLowerCase()

  // ── Filter papers: no bad titles, no duplicates, exclude current paper ────
  const seenTitles = new Set<string>()
  const papers = (state.related_papers || [])
    .filter(p => {
      if (!p.url || BAD_LABEL(p.title)) return false
      const key = p.title.trim().toLowerCase()
      if (seenTitles.has(key)) return false
      if (paperTitleLower && key === paperTitleLower) return false  // exclude self
      seenTitles.add(key)
      return true
    })
    .slice(0, 10)

  // ── Filter user interest topics: remove garbage ───────────────────────────
  const topics = (state.user_interest_topics || [])
    .filter(t => !BAD_LABEL(t) && !JUNK_WORDS.has(t.toLowerCase().trim()))
    .slice(0, 5)

  // ── Filter concepts: multi-word preferred, no bare stopwords ──────────────
  const concepts = (state.key_concepts || [])
    .filter(c => c && c.trim().length >= 4 && !JUNK_WORDS.has(c.toLowerCase().trim()))
    .slice(0, 6)

  const methods = (state.methods || []).filter(Boolean).slice(0, 4)

  const root: MindNode = { id: 'root', label: 'You', type: 'root', children: [] }
  if (papers.length === 0 && concepts.length === 0) return root

  // ─ Determine L1 (domains) ─────────────────────────────────────────────────
  // User interest topics (if good), otherwise paper title as single domain
  const hasTopics = topics.length >= 2  // need at least 2 meaningful topics
  const l1Labels = hasTopics
    ? topics
    : [paperTitle || 'Current Paper']

  // ─ Determine L2 (subtopics) ───────────────────────────────────────────────
  const l2Labels = [...concepts, ...methods].slice(0, 8)

  // ─ Build tree ─────────────────────────────────────────────────────────────
  l1Labels.forEach((domain, di) => {
    const domainNode: MindNode = {
      id: `l1-${di}`, label: domain, type: 'category', category: 'domain', children: [],
    }

    if (hasTopics) {
      // Multi-domain mode: distribute subtopics across domains
      const myL2 = l2Labels.filter((_, si) => si % l1Labels.length === di)
      myL2.forEach((sub, si) => {
        const subNode: MindNode = {
          id: `l2-${di}-${si}`, label: sub, type: 'category', category: 'concept', children: [],
        }
        // Assign papers round-robin across all subtopics
        const globalIdx = l2Labels.indexOf(sub)
        papers
          .filter((_, pi) => pi % l2Labels.length === globalIdx)
          .forEach((p, pi) => subNode.children!.push({
            id: `p-${di}-${si}-${pi}`, label: p.title, type: 'leaf', category: 'paper', url: p.url,
          }))
        if (subNode.children!.length > 0) domainNode.children!.push(subNode)
      })
      // Any papers not assigned via subtopics → attach directly
      if (domainNode.children!.length === 0) {
        papers.filter((_, pi) => pi % l1Labels.length === di).forEach((p, pi) =>
          domainNode.children!.push({
            id: `p-${di}-${pi}`, label: p.title, type: 'leaf', category: 'paper', url: p.url,
          }))
      }
    } else {
      // Single-domain mode: all subtopics go under the paper title
      if (l2Labels.length > 0 && papers.length > 0) {
        l2Labels.forEach((sub, si) => {
          const subNode: MindNode = {
            id: `l2-0-${si}`, label: sub, type: 'category', category: 'concept', children: [],
          }
          papers
            .filter((_, pi) => pi % l2Labels.length === si)
            .forEach((p, pi) => subNode.children!.push({
              id: `p-0-${si}-${pi}`, label: p.title, type: 'leaf', category: 'paper', url: p.url,
            }))
          if (subNode.children!.length > 0) domainNode.children!.push(subNode)
        })
      } else {
        // No subtopics — papers directly under domain
        papers.forEach((p, pi) => domainNode.children!.push({
          id: `p-0-${pi}`, label: p.title, type: 'leaf', category: 'paper', url: p.url,
        }))
      }
    }

    if (domainNode.children!.length > 0) root.children!.push(domainNode)
  })

  return root
}

function truncateLabel(label: string, maxLen: number): string {
  return label.length <= maxLen ? label : label.slice(0, maxLen - 1) + '…'
}

function MindMap({ state, onAsk }: { state: AnalysisState; onAsk: (q: string) => void }) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [tooltip, setTooltip] = useState<{ text: string; x: number; y: number } | null>(null)
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set())
  const containerRef = useRef<HTMLDivElement>(null)
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null)
  const svgSelRef = useRef<d3.Selection<SVGSVGElement, unknown, null, undefined> | null>(null)
  const initialTransformRef = useRef<d3.ZoomTransform | null>(null)

  // Only render the tree when we have papers (the leaves) — not just concepts
  const goodPapers = (state.related_papers || []).filter(p => p.url && !BAD_LABEL(p.title))
  const hasData = goodPapers.length > 0

  useEffect(() => {
    if (!hasData || !svgRef.current || !containerRef.current) return

    const container = containerRef.current
    const W = container.clientWidth  || 380
    const H = container.clientHeight || 480

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()
    const g = svg.append('g')

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.2, 3])
      .on('zoom', (event) => { g.attr('transform', event.transform) })
    svg.call(zoom)
    zoomRef.current = zoom
    svgSelRef.current = svg

    function filterCollapsed(node: MindNode): any {
      if (collapsed.has(node.id)) return { ...node, children: undefined }
      return { ...node, children: node.children?.map(filterCollapsed) }
    }

    const treeData = filterCollapsed(buildMindTree(state))
    const root = d3.hierarchy(treeData)

    // Horizontal left-to-right layout: d.y = depth (→), d.x = breadth (↕)
    const NODE_W = 130  // node box width for leaves / categories
    const NODE_H = 26   // node box height
    const DEPTH_GAP   = 155  // horizontal gap between levels
    const BREADTH_GAP = 38   // minimum vertical gap between siblings

    const treeLayout = d3.tree<any>().nodeSize([BREADTH_GAP, DEPTH_GAP])
    treeLayout(root)

    const allNodes = root.descendants()
    const minBreadth = d3.min(allNodes, (d: any) => d.x) ?? 0
    const maxBreadth = d3.max(allNodes, (d: any) => d.x) ?? 0
    const maxDepth   = d3.max(allNodes, (d: any) => d.y) ?? 0

    // Scale to fit inside the viewport, capped at 1×
    const treeW = maxDepth + NODE_W + 20
    const treeH = maxBreadth - minBreadth + BREADTH_GAP
    const scale  = Math.min(1, (W - 20) / treeW, (H - 20) / treeH)

    // Translate so the whole tree is centred in the viewport
    const tx = 10
    const ty = H / 2 - (minBreadth + maxBreadth) / 2 * scale

    // node helpers (in raw tree coords, transform applied by zoom)
    const nx = (d: any) => (d as any).y   // horizontal pos (depth)
    const ny = (d: any) => (d as any).x   // vertical pos (breadth)

    const linkGen = d3.linkHorizontal<any, any>()
      .x((d: any) => nx(d))
      .y((d: any) => ny(d))

    g.selectAll('.link').data(root.links()).join('path')
      .attr('class', 'link').attr('d', linkGen).attr('fill', 'none')
      .attr('stroke', (d: any) => CATEGORY_COLORS[d.target.data.category || 'root'] || '#3f3f46')
      .attr('stroke-width', 1.2).attr('stroke-opacity', 0.4)

    const node = g.selectAll('.node').data(allNodes).join('g')
      .attr('class', 'node')
      .attr('transform', (d: any) => `translate(${nx(d)},${ny(d)})`)
      .style('cursor', 'pointer')

    node.each(function (d: any) {
      const sel  = d3.select(this)
      const data: MindNode = d.data
      const isRoot  = data.type === 'root'
      const isCat   = data.type === 'category'
      const isPaper = data.category === 'paper'
      const color   = CATEGORY_COLORS[data.category || 'root'] || '#3f3f46'
      const w = isRoot ? 52 : NODE_W
      const h = isRoot ? 30 : NODE_H
      const label = truncateLabel(data.label, isRoot ? 8 : isPaper ? 17 : 14)
      const isCollapsedNode = collapsed.has(data.id)

      // Glow halo for root only
      if (isRoot) {
        sel.append('rect').attr('class', 'halo')
          .attr('x', -w / 2 - 4).attr('y', -h / 2 - 4)
          .attr('width', w + 8).attr('height', h + 8)
          .attr('rx', 13).attr('fill', 'rgba(59,130,246,0.1)').attr('stroke', 'none')
      }

      // Main border rect — class 'box' for hover targeting
      sel.append('rect').attr('class', 'box')
        .attr('x', -w / 2).attr('y', -h / 2).attr('width', w).attr('height', h)
        .attr('rx', isRoot ? 10 : isCat ? 7 : 5)
        .attr('fill', isRoot ? '#1a1a2e' : isPaper ? `${color}15` : `${color}18`)
        .attr('stroke', isRoot ? '#3b82f6' : color)
        .attr('stroke-width', isRoot ? 2 : isCat ? 1.5 : 1)
        .attr('stroke-opacity', isRoot ? 1 : 0.5)
        .attr('stroke-dasharray', isPaper ? '4,2' : 'none')

      // Collapse indicator for category nodes that have children
      if (isCat && d.children) {
        sel.append('circle').attr('cx', w / 2 - 7).attr('cy', 0).attr('r', 4)
          .attr('fill', isCollapsedNode ? color : '#1f1f23').attr('stroke', color).attr('stroke-width', 1)
        sel.append('text').attr('x', w / 2 - 7).attr('y', 3.5).attr('text-anchor', 'middle')
          .attr('font-size', '7px').attr('fill', isCollapsedNode ? '#fff' : color)
          .attr('pointer-events', 'none').text(isCollapsedNode ? '+' : '−')
      }

      // Link arrow icon for paper leaves
      if (isPaper) {
        sel.append('text').attr('x', -w / 2 + 9).attr('y', 0.5)
          .attr('dominant-baseline', 'middle').attr('font-size', '9px')
          .attr('fill', color).attr('opacity', 0.75).attr('pointer-events', 'none').text('↗')
      }

      // Label text
      sel.append('text').attr('x', isPaper ? 5 : 0)
        .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
        .attr('fill', isRoot ? '#e4e4e7' : isPaper ? color : isCat ? color : '#a1a1aa')
        .attr('font-size', isRoot ? '11px' : isCat ? '10px' : '9.5px')
        .attr('font-weight', isRoot ? '700' : isCat ? '600' : '400')
        .attr('text-decoration', isPaper ? 'underline' : 'none')
        .attr('pointer-events', 'none').text(label)
    })

    node.on('mouseenter', function (event, d: any) {
      const data: MindNode = d.data
      if (data.label.length > 14) {
        const rect = container.getBoundingClientRect()
        setTooltip({ text: data.label, x: event.clientX - rect.left, y: event.clientY - rect.top })
      }
      d3.select(this).select('rect.box')
        .attr('stroke-opacity', 1)
        .attr('stroke-width', data.type === 'root' ? 2.5 : 2)
    }).on('mouseleave', function (_e, d: any) {
      setTooltip(null)
      const data: MindNode = d.data
      d3.select(this).select('rect.box')
        .attr('stroke-opacity', data.type === 'root' ? 1 : 0.5)
        .attr('stroke-width', data.type === 'root' ? 2 : data.type === 'category' ? 1.5 : 1)
    }).on('click', function (event, d: any) {
      event.stopPropagation()
      const data: MindNode = d.data
      if (data.type === 'category' && d.data.children) {
        setCollapsed(prev => {
          const n = new Set(prev)
          n.has(data.id) ? n.delete(data.id) : n.add(data.id)
          return n
        })
      } else if (data.category === 'paper' && data.url) {
        window.open(data.url, '_blank')
      }
    })

    // Fit the full tree into the viewport on initial render
    const initialTransform = d3.zoomIdentity.translate(tx, ty).scale(scale)
    initialTransformRef.current = initialTransform
    svg.call(zoom.transform, initialTransform)
  }, [state, collapsed])

  const resetZoom = () => {
    if (zoomRef.current && svgSelRef.current && initialTransformRef.current)
      svgSelRef.current.transition().duration(350).call(zoomRef.current.transform, initialTransformRef.current)
  }

  if (!hasData) {
    return (
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12 }}>
        <svg width="44" height="44" viewBox="0 0 44 44" fill="none" style={{ opacity: 0.2 }}>
          <circle cx="22" cy="11" r="6" stroke="#a1a1aa" strokeWidth="1.5" />
          <circle cx="9"  cy="33" r="6" stroke="#a1a1aa" strokeWidth="1.5" />
          <circle cx="35" cy="33" r="6" stroke="#a1a1aa" strokeWidth="1.5" />
          <path d="M22 17v7M22 24L10 29M22 24l12 5" stroke="#a1a1aa" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
        <p style={{ margin: 0, fontSize: 11.5, color: '#2d2d30', textAlign: 'center', lineHeight: 1.7 }}>
          {state.status === 'idle'
            ? 'Analyse a paper to generate\nthe knowledge map'
            : state.status === 'running' || state.status === 'partial'
            ? 'Fetching citations and\nbuilding knowledge map…'
            : 'No cited papers found yet.\nTry analysing a paper.'}
        </p>
      </div>
    )
  }

  return (
    <div ref={containerRef} style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
      {/* Controls */}
      <div style={{ position: 'absolute', top: 8, left: 8, zIndex: 10 }}>
        <button onClick={resetZoom} title="Reset zoom to fit"
          style={{ padding: '3px 9px', borderRadius: 6, background: '#111113', border: '1px solid #1f1f23', color: '#52525b', fontSize: 10, cursor: 'pointer', fontFamily: 'inherit', transition: 'all 0.15s' }}
          onMouseEnter={e => { e.currentTarget.style.color = '#a1a1aa'; e.currentTarget.style.borderColor = '#2d2d30' }}
          onMouseLeave={e => { e.currentTarget.style.color = '#52525b'; e.currentTarget.style.borderColor = '#1f1f23' }}>
          ⊡ Fit
        </button>
      </div>
      {/* Legend */}
      <div style={{ position: 'absolute', top: 8, right: 8, zIndex: 10, display: 'flex', flexDirection: 'column', gap: 3 }}>
        {Object.entries({ concepts: 'Concepts', methods: 'Methods', related: 'Related', gaps: 'Gaps', ideas: 'Ideas' }).map(([k, v]) => (
          <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <div style={{ width: 8, height: 8, borderRadius: 2, background: CATEGORY_COLORS[k] }} />
            <span style={{ fontSize: 9, color: '#3f3f46' }}>{v}</span>
          </div>
        ))}
      </div>
      {/* Hint */}
      <div style={{ position: 'absolute', bottom: 8, left: 8, zIndex: 10, fontSize: 9, color: '#1f1f23', lineHeight: 1.7 }}>
        Scroll · zoom &nbsp;·&nbsp; Drag · pan &nbsp;·&nbsp; Click category &nbsp;·&nbsp; Click leaf → ask
      </div>

      <svg ref={svgRef} width="100%" height="100%" style={{ background: '#09090b' }} />

      {tooltip && (
        <div style={{ position: 'absolute', left: tooltip.x + 8, top: tooltip.y - 8, zIndex: 20, background: '#18181b', border: '1px solid #2d2d30', borderRadius: 8, padding: '6px 10px', fontSize: 11, color: '#d4d4d8', maxWidth: 220, pointerEvents: 'none', lineHeight: 1.5 }}>
          {tooltip.text}
        </div>
      )}
    </div>
  )
}

// ── App Shell ─────────────────────────────────────────────────────────────────
export default function App() {
  const [tabId, setTabId] = useState<number | null>(null)
  const [state, setState] = useState<AnalysisState>(IS_EXTENSION ? DEFAULT_STATE : MOCK_STATE)
  const [view, setView] = useState<View>('chat')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [thinking, setThinking] = useState(false)
  const chatBottomRef = useRef<HTMLDivElement>(null)
  const lastAnswerKey = useRef('')

  useEffect(() => {
    loadConfig()
    notifySidebarReady()
    getCurrentTabId().then(async tid => {
      setTabId(tid)
      if (tid != null) {
        try { setState(await getTabState(tid)) } catch {}
      }
    })
  }, [])

  useEffect(() => {
    if (tabId == null) return
    return onStateUpdate(tabId, (newState) => {
      setState(newState)
      if (newState.chat_status === 'complete' && newState.chat_answer) {
        const key = `${newState.chat_question}|||${newState.chat_answer}`
        if (key === lastAnswerKey.current) return
        lastAnswerKey.current = key
        setThinking(false)
        setMessages(prev => [...prev, {
          id: nextId(), role: 'assistant', text: newState.chat_answer, typing: true,
          followups: newState.chat_follow_up_questions?.slice(0, 3),
          citations: newState.chat_citations?.slice(0, 3) as any,
        }])
      }
      if (newState.chat_status === 'failed') {
        const key = `err|||${newState.chat_error}`
        if (key === lastAnswerKey.current) return
        lastAnswerKey.current = key
        setThinking(false)
        setMessages(prev => [...prev, {
          id: nextId(), role: 'assistant',
          text: newState.chat_error || 'Something went wrong. Is the backend running on localhost:8000?',
        }])
      }
    })
  }, [tabId])

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, thinking])

  const send = useCallback(async (q: string) => {
    const question = q.trim()
    if (!question || thinking) return
    setInput('')
    lastAnswerKey.current = ''
    setMessages(prev => [...prev, { id: nextId(), role: 'user', text: question }])
    setThinking(true)
    setView('chat')

    if (!IS_EXTENSION) {
      await new Promise(r => setTimeout(r, 800))
      const ans = question.toLowerCase().includes('limit')
        ? 'The primary limitation is the **quadratic O(n²) complexity** of self-attention with respect to sequence length. This makes the Transformer impractical for very long sequences without modifications.\n\nThe paper trains on sequences up to 512 tokens and notes memory constraints. Additionally:\n- No recurrent structure — positional encoding is the sole mechanism for sequence order\n- May not generalize to sequence lengths unseen during training\n- Fixed context window unlike RNN variants'
        : 'The Transformer uses **scaled dot-product attention**: queries, keys, and values are linearly projected, attention weights computed as `softmax(QKᵀ/√dₖ)`, then applied to values.\n\n**Multi-head attention** runs h=8 parallel heads with dₖ=dᵥ=64, whose outputs are concatenated and projected. Key benefits:\n- Replaces recurrence entirely, enabling full parallelization\n- O(1) sequential operations (vs O(n) for RNNs)\n- Each head can attend to different representation subspaces'
      setThinking(false)
      setMessages(prev => [...prev, {
        id: nextId(), role: 'assistant', text: ans, typing: true,
        followups: [
          'What was the training time and hardware for the base vs big Transformer?',
          'How does multi-head attention compare to single-head with full dₘₒₐₑₗ dimension?',
          'What positional encoding scheme does the paper use?',
        ],
        citations: MOCK_STATE.related_papers.slice(0, 2) as any,
      }])
      return
    }

    try {
      if (tabId != null) await askChat(tabId, question)
    } catch {
      setThinking(false)
      lastAnswerKey.current = 'err|||catch'
      setMessages(prev => [...prev, {
        id: nextId(), role: 'assistant',
        text: 'Could not reach the backend. Make sure it is running on `localhost:8000`.',
      }])
    }
  }, [thinking, tabId])

  const hasPaper = !!(state.paper_title || state.pageUrl)
  const isAnalyzing = state.status === 'running' || state.status === 'partial'
  const paperLabel = state.paper_title || (state.pageUrl ? 'Analysing…' : '')

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: '#09090b', overflow: 'hidden', fontFamily: "-apple-system,'Inter','Segoe UI',system-ui,sans-serif", color: '#e4e4e7' }}>

      {/* ── Header ── */}
      <div style={{ flexShrink: 0, background: '#09090b', borderBottom: '1px solid #111113' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 14px 6px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
              <circle cx="9" cy="9" r="7.5" stroke="#3b82f6" strokeWidth="1.5" />
              <circle cx="9" cy="9" r="3" fill="#3b82f6" opacity="0.35" />
              <path d="M9 4v5M9 9l3.5 2.5" stroke="#3b82f6" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
            <span style={{ fontSize: 13, fontWeight: 700, color: '#fafafa', letterSpacing: '-0.02em' }}>Research Copilot</span>
          </div>

          <button
            onClick={() => { if (tabId != null) triggerAnalysis(tabId).catch(() => {}) }}
            style={{ fontSize: 10.5, padding: '4px 10px', borderRadius: 6, background: isAnalyzing ? 'rgba(245,158,11,0.1)' : 'rgba(59,130,246,0.06)', border: `1px solid ${isAnalyzing ? 'rgba(245,158,11,0.4)' : '#27272a'}`, color: isAnalyzing ? '#f59e0b' : '#52525b', cursor: 'pointer', fontFamily: 'inherit', transition: 'all 0.15s', display: 'flex', alignItems: 'center', gap: 5 }}
            onMouseEnter={e => { if (!isAnalyzing) { e.currentTarget.style.borderColor = 'rgba(59,130,246,0.3)'; e.currentTarget.style.color = '#71717a' } }}
            onMouseLeave={e => { if (!isAnalyzing) { e.currentTarget.style.borderColor = '#27272a'; e.currentTarget.style.color = '#52525b' } }}>
            {isAnalyzing ? (
              <><span style={{ width: 6, height: 6, borderRadius: '50%', background: '#f59e0b', display: 'inline-block', animation: 'pulse 1.2s ease-in-out infinite' }} /> Analysing…</>
            ) : (
              <><svg width="10" height="10" viewBox="0 0 10 10" fill="none" style={{ flexShrink: 0 }}>
                <path d="M9 5a4 4 0 11-1.4-3" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
                <path d="M9 2v3H6" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
              </svg> Analyse</>
            )}
          </button>
        </div>

        {/* Paper title + status icon */}
        {paperLabel && (
          <div style={{ padding: '0 14px 6px', display: 'flex', alignItems: 'center', gap: 6 }}>
            <svg width="10" height="10" viewBox="0 0 10 10" fill="none" style={{ flexShrink: 0, opacity: 0.35 }}>
              <rect x="1" y="1" width="8" height="8" rx="1.5" stroke="#a1a1aa" strokeWidth="1.2" />
              <path d="M3 4h4M3 6h3" stroke="#a1a1aa" strokeWidth="1" strokeLinecap="round" />
            </svg>
            <p style={{ margin: 0, fontSize: 10.5, color: '#3f3f46', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
              {paperLabel}
            </p>
            {state.status === 'complete' && (
              <svg width="10" height="10" viewBox="0 0 10 10" fill="none" style={{ flexShrink: 0 }}>
                <circle cx="5" cy="5" r="4" stroke="#22c55e" strokeWidth="1.2" />
                <path d="M2.5 5l1.5 1.5 3-3" stroke="#22c55e" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            )}
            {state.status === 'failed' && (
              <svg width="10" height="10" viewBox="0 0 10 10" fill="none" style={{ flexShrink: 0 }}>
                <circle cx="5" cy="5" r="4" stroke="#ef4444" strokeWidth="1.2" />
                <path d="M3.5 3.5l3 3M6.5 3.5l-3 3" stroke="#ef4444" strokeWidth="1.2" strokeLinecap="round" />
              </svg>
            )}
          </div>
        )}

        <StatusStrip state={state} />

        {/* Tabs */}
        <div style={{ display: 'flex', padding: '0 14px' }}>
          {(['chat', 'graph'] as View[]).map(v => (
            <button key={v} onClick={() => setView(v)}
              style={{ flex: 1, padding: '7px 0', background: 'transparent', border: 'none', borderBottom: `2px solid ${view === v ? '#3b82f6' : 'transparent'}`, color: view === v ? '#e4e4e7' : '#3f3f46', fontSize: 12, fontWeight: view === v ? 600 : 400, cursor: 'pointer', fontFamily: 'inherit', transition: 'all 0.15s', letterSpacing: '0.01em' }}>
              {v === 'graph' ? '🗺 Map' : '💬 Chat'}
            </button>
          ))}
        </div>
      </div>

      {/* ── Views ── */}
      <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {view === 'chat' && (
          <ChatView
            messages={messages} thinking={thinking}
            input={input} onInput={setInput} onSend={send}
            hasPaper={hasPaper} bottomRef={chatBottomRef}
            paperTitle={state.paper_title}
            keyConcepts={state.key_concepts}
          />
        )}
        {view === 'graph' && (
          <MindMap state={state} onAsk={q => { setView('chat'); send(q) }} />
        )}
      </div>

      <style>{`
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
        @keyframes dot-bounce { 0%,80%,100%{transform:translateY(0)} 40%{transform:translateY(-5px)} }
        .dot-1{animation:dot-bounce 1.2s infinite .0s}
        .dot-2{animation:dot-bounce 1.2s infinite .2s}
        .dot-3{animation:dot-bounce 1.2s infinite .4s}
        ::-webkit-scrollbar{width:4px}
        ::-webkit-scrollbar-track{background:transparent}
        ::-webkit-scrollbar-thumb{background:#1f1f23;border-radius:4px}
        ::-webkit-scrollbar-thumb:hover{background:#27272a}
      `}</style>
    </div>
  )
}

// ── Chat View ─────────────────────────────────────────────────────────────────
function ChatView({ messages, thinking, input, onInput, onSend, hasPaper, bottomRef, paperTitle, keyConcepts }: {
  messages: ChatMessage[]; thinking: boolean; input: string
  onInput: (s: string) => void; onSend: (s: string) => void
  hasPaper: boolean; bottomRef: React.RefObject<HTMLDivElement>
  paperTitle?: string; keyConcepts?: string[]
}) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = Math.min(ta.scrollHeight, 120) + 'px'
  }, [input])

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSend(input) }
  }

  const starterQuestions = keyConcepts && keyConcepts.length > 0 ? [
    'What is the core contribution and novelty of this paper?',
    `How does ${keyConcepts[0]} work technically?`,
    'What are the key experimental results and how do they compare to baselines?',
    'What are the main limitations and open problems identified?',
  ] : [
    'What is the core contribution and novelty of this paper?',
    'How does the proposed method work technically?',
    'What are the key results and benchmarks?',
    'What limitations does the paper acknowledge?',
  ]

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{ flex: 1, overflowY: 'auto', padding: '14px 14px 8px', display: 'flex', flexDirection: 'column', gap: 16 }}>
        {messages.length === 0 && !thinking && (
          <div style={{ display: 'flex', flexDirection: 'column', flex: 1, gap: 10, paddingBottom: 8 }}>
            {hasPaper ? (
              <>
                {paperTitle && (
                  <div style={{ padding: '10px 12px', borderRadius: 10, background: 'rgba(59,130,246,0.06)', border: '1px solid rgba(59,130,246,0.12)' }}>
                    <p style={{ margin: 0, fontSize: 11, color: '#60a5fa', fontWeight: 500, lineHeight: 1.5 }}>{paperTitle}</p>
                  </div>
                )}
                <p style={{ margin: '4px 0 2px', fontSize: 10, color: '#27272a', textTransform: 'uppercase', letterSpacing: '0.07em' }}>Suggested questions</p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                  {starterQuestions.map(q => (
                    <button key={q} onClick={() => onSend(q)}
                      style={{ padding: '9px 13px', borderRadius: 10, background: '#111113', border: '1px solid #1a1a1e', color: '#52525b', fontSize: 11.5, cursor: 'pointer', textAlign: 'left', fontFamily: 'inherit', transition: 'all 0.15s', lineHeight: 1.5, display: 'flex', alignItems: 'flex-start', gap: 7 }}
                      onMouseEnter={e => { e.currentTarget.style.background = '#18181b'; e.currentTarget.style.color = '#a1a1aa'; e.currentTarget.style.borderColor = '#2d2d30' }}
                      onMouseLeave={e => { e.currentTarget.style.background = '#111113'; e.currentTarget.style.color = '#52525b'; e.currentTarget.style.borderColor = '#1a1a1e' }}>
                      <span style={{ color: '#2d2d30', flexShrink: 0, marginTop: 1 }}>↗</span>
                      <span>{q}</span>
                    </button>
                  ))}
                </div>
              </>
            ) : (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12, paddingBottom: 40 }}>
                <svg width="42" height="42" viewBox="0 0 42 42" fill="none" style={{ opacity: 0.2 }}>
                  <rect x="5" y="7" width="32" height="28" rx="4" stroke="#a1a1aa" strokeWidth="1.5" />
                  <path d="M13 16h16M13 21h12M13 26h8" stroke="#a1a1aa" strokeWidth="1.5" strokeLinecap="round" />
                </svg>
                <p style={{ margin: 0, fontSize: 12, color: '#27272a', textAlign: 'center', lineHeight: 1.9 }}>
                  Open a paper on arXiv,<br />Semantic Scholar, or OpenReview<br />
                  <span style={{ color: '#3f3f46' }}>then click</span> <span style={{ color: '#52525b', fontWeight: 500 }}>↺ Analyse</span>
                </p>
              </div>
            )}
          </div>
        )}

        {messages.map((msg, i) => (
          <Bubble key={msg.id} msg={msg} isLatest={i === messages.length - 1} onFollowup={onSend} />
        ))}

        {thinking && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '4px 0' }}>
            <div style={{ display: 'flex', gap: 4, padding: '10px 14px', background: '#0d0d10', borderRadius: '4px 14px 14px 14px', border: '1px solid #1a1a1e' }}>
              {[1, 2, 3].map(i => (
                <div key={i} className={`dot-${i}`} style={{ width: 5, height: 5, borderRadius: '50%', background: '#3b82f6', opacity: 0.6 }} />
              ))}
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div style={{ flexShrink: 0, padding: '8px 12px 14px', borderTop: '1px solid #111113' }}>
        <div
          style={{ display: 'flex', alignItems: 'flex-end', gap: 8, background: '#0d0d10', border: '1px solid #1a1a1e', borderRadius: 14, padding: '10px 12px', transition: 'border-color 0.15s' }}
          onFocusCapture={e => e.currentTarget.style.borderColor = 'rgba(59,130,246,0.4)'}
          onBlurCapture={e => e.currentTarget.style.borderColor = '#1a1a1e'}>
          <textarea ref={textareaRef} value={input} onChange={e => onInput(e.target.value)} onKeyDown={onKey}
            placeholder={hasPaper ? 'Ask anything about this paper…' : 'Open a paper first…'}
            disabled={!hasPaper || thinking} rows={1}
            style={{ flex: 1, background: 'transparent', border: 'none', outline: 'none', color: '#e4e4e7', fontSize: 13, fontFamily: 'inherit', resize: 'none', lineHeight: 1.5, overflowY: 'auto', opacity: !hasPaper ? 0.3 : 1 }} />
          <button onClick={() => onSend(input)} disabled={!input.trim() || !hasPaper || thinking}
            style={{ flexShrink: 0, width: 30, height: 30, borderRadius: 8, border: 'none', background: input.trim() && hasPaper && !thinking ? '#3b82f6' : '#111113', color: input.trim() && hasPaper && !thinking ? '#fff' : '#27272a', cursor: input.trim() && hasPaper && !thinking ? 'pointer' : 'default', display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'all 0.15s' }}>
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
              <path d="M10.5 6L1 1.5L3 6L1 10.5L10.5 6Z" fill="currentColor" />
            </svg>
          </button>
        </div>
        <p style={{ margin: '4px 0 0 2px', fontSize: 9.5, color: '#1f1f23' }}>Shift+Enter for new line</p>
      </div>
    </div>
  )
}

// ── Chat Bubble ───────────────────────────────────────────────────────────────
function Bubble({ msg, isLatest, onFollowup }: { msg: ChatMessage; isLatest: boolean; onFollowup: (q: string) => void }) {
  const [typed, setTyped] = useState(!msg.typing)

  if (msg.role === 'user') {
    return (
      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <div style={{ maxWidth: '85%', background: '#1a1a1e', borderRadius: '14px 14px 4px 14px', padding: '10px 14px', color: '#d4d4d8', fontSize: 13, lineHeight: 1.6 }}>
          {msg.text}
        </div>
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {/* Answer */}
      <div style={{ background: '#0d0d10', borderRadius: '4px 14px 14px 14px', padding: '12px 14px', border: '1px solid #1a1a1e' }}>
        <div style={{ color: '#d4d4d8', fontSize: 13, lineHeight: 1.75 }}>
          {msg.typing && !typed
            ? <TypewriterText text={msg.text} onDone={() => setTyped(true)} />
            : renderMarkdown(msg.text)}
        </div>
        {typed && (
          <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 6, borderTop: '1px solid #111113', paddingTop: 5 }}>
            <CopyBtn text={msg.text} />
          </div>
        )}
      </div>

      {/* Citations */}
      {typed && msg.citations && msg.citations.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          <p style={{ margin: '0 0 2px', fontSize: 9.5, color: '#27272a', textTransform: 'uppercase', letterSpacing: '0.07em' }}>Sources</p>
          {msg.citations.map((p, i) => (
            <a key={i} href={(p as any).url} target="_blank" rel="noreferrer"
              style={{ display: 'flex', alignItems: 'center', gap: 7, padding: '6px 10px', borderRadius: 8, background: '#0d0d0f', border: '1px solid #1a1a1e', textDecoration: 'none', transition: 'border-color 0.15s' }}
              onMouseEnter={e => e.currentTarget.style.borderColor = '#2d2d30'}
              onMouseLeave={e => e.currentTarget.style.borderColor = '#1a1a1e'}>
              <svg width="9" height="9" viewBox="0 0 9 9" fill="none" style={{ flexShrink: 0, opacity: 0.45 }}>
                <path d="M1.5 1.5h4v4M5.5 1.5L1 6" stroke="#52525b" strokeWidth="1.2" strokeLinecap="round" />
              </svg>
              <span style={{ fontSize: 11, color: '#3f3f46', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.title}</span>
            </a>
          ))}
        </div>
      )}

      {/* Follow-up questions */}
      {typed && isLatest && msg.followups && msg.followups.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <p style={{ margin: '0 0 2px', fontSize: 9.5, color: '#27272a', textTransform: 'uppercase', letterSpacing: '0.07em' }}>Follow up</p>
          {msg.followups.map((q, i) => (
            <button key={i} onClick={() => onFollowup(q)}
              style={{ textAlign: 'left', padding: '8px 12px', borderRadius: 10, background: 'rgba(59,130,246,0.04)', border: '1px solid rgba(59,130,246,0.1)', color: '#4d7fb8', fontSize: 11.5, cursor: 'pointer', fontFamily: 'inherit', transition: 'all 0.15s', lineHeight: 1.5, display: 'flex', gap: 6 }}
              onMouseEnter={e => { e.currentTarget.style.background = 'rgba(59,130,246,0.09)'; e.currentTarget.style.borderColor = 'rgba(59,130,246,0.25)'; e.currentTarget.style.color = '#60a5fa' }}
              onMouseLeave={e => { e.currentTarget.style.background = 'rgba(59,130,246,0.04)'; e.currentTarget.style.borderColor = 'rgba(59,130,246,0.1)'; e.currentTarget.style.color = '#4d7fb8' }}>
              <span style={{ opacity: 0.5, flexShrink: 0 }}>↗</span>
              <span>{q}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
