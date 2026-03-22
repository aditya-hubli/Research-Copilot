export const IS_EXTENSION =
  typeof window !== 'undefined' &&
  !!(window as any).chrome?.runtime?.id

export interface Paper { title: string; url: string; score?: number }

export interface AnalysisState {
  pageUrl: string
  paper_title: string
  status: 'idle' | 'running' | 'partial' | 'complete' | 'failed'
  summary: string
  key_concepts: string[]
  methods: string[]
  datasets: string[]
  related_papers: Paper[]
  research_gaps: string[]
  ideas: string[]
  error: string
  chat_status: 'idle' | 'running' | 'complete' | 'failed'
  chat_question: string
  chat_answer: string
  chat_error: string
  chat_citations: Paper[]
  chat_evidence_snippets: Array<{ title: string; url: string; section?: string; snippet: string }>
  chat_follow_up_questions: string[]
  // Runtime-only fields from background worker
  confidence?: number
  from_cache?: boolean
  current_paper_indexed?: boolean
  current_paper_chunks?: number
  current_paper_indexing?: boolean
  provider?: string
  research_connections?: string[]
  user_interest_topics?: string[]
  autonomy_notes?: string[]
  job_id?: string
  resolved_url?: string
  source?: string
  related_papers_provider?: string
  related_papers_loading?: boolean
  review_required?: boolean
  review_id?: string
}

export type LLMProvider = 'openai' | 'anthropic' | 'gemini' | 'huggingface'

export interface ExtConfig {
  backendApiBase: string
  backendApiKey: string
  llmApiKey: string
  llmProvider: LLMProvider
  userId: string
}

export const DEFAULT_STATE: AnalysisState = {
  pageUrl: '', paper_title: '', status: 'idle', summary: '',
  key_concepts: [], methods: [], datasets: [], related_papers: [],
  research_gaps: [], ideas: [], error: '',
  chat_status: 'idle', chat_question: '', chat_answer: '', chat_error: '',
  chat_citations: [], chat_evidence_snippets: [], chat_follow_up_questions: [],
}

export const MOCK_STATE: AnalysisState = {
  pageUrl: 'https://arxiv.org/abs/2203.02155',
  paper_title: 'Training language models to follow instructions with human feedback',
  status: 'complete',
  summary: 'Introduces InstructGPT — fine-tuning GPT-3 via RLHF to follow user intent. Uses supervised fine-tuning on human demonstrations, reward model training on human preference rankings, and PPO-based RL. InstructGPT-1.3B is preferred over GPT-3-175B by labelers, demonstrating alignment gains far exceed scale gains.',
  key_concepts: ['Reinforcement Learning from Human Feedback', 'Instruction Following', 'Reward Modeling', 'Policy Optimization', 'Alignment Tax'],
  methods: ['Supervised Fine-Tuning (SFT)', 'Reward Model Training', 'PPO with KL Penalty', 'Human Preference Ranking'],
  datasets: ['InstructGPT Prompt Dataset', 'OpenAI API Prompts', 'Human-Written Demonstrations'],
  related_papers: [
    { title: 'Constitutional AI: Harmlessness from AI Feedback', url: 'https://arxiv.org/abs/2212.08073', score: 0.93 },
    { title: 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model', url: 'https://arxiv.org/abs/2305.18290', score: 0.89 },
    { title: 'Learning to summarize from human feedback', url: 'https://arxiv.org/abs/2009.01325', score: 0.84 },
    { title: 'Proximal Policy Optimization Algorithms', url: 'https://arxiv.org/abs/1707.06347', score: 0.79 },
  ],
  research_gaps: [
    'The KL penalty coefficient is tuned manually and may not generalize across domains — an adaptive KL schedule could improve stability during RL fine-tuning.',
    'Labeler agreement is assumed uniform but individual annotator biases are not modeled — a mixture-of-annotators reward model could capture preference diversity.',
    'InstructGPT is evaluated primarily on English prompts; multilingual alignment under RLHF with cross-lingual preference data remains unexplored.',
  ],
  ideas: [
    'Extend RLHF with per-annotator reward models that capture individual value systems, enabling personalized alignment without collapsing diverse preferences into a single reward signal.',
    'Apply the SFT + RM + PPO pipeline to code generation with execution-based rewards, replacing human preference labels with test-pass signals for objective preference estimation.',
  ],
  error: '',
  chat_status: 'idle', chat_question: '', chat_answer: '', chat_error: '',
  chat_citations: [], chat_evidence_snippets: [], chat_follow_up_questions: [],
}

function sendMsg(msg: Record<string, unknown>): Promise<any> {
  return new Promise((resolve, reject) => {
    const chrome = (window as any).chrome
    chrome.runtime.sendMessage(msg, (res: any) => {
      if (chrome.runtime.lastError) { reject(new Error(chrome.runtime.lastError.message)); return }
      if (!res?.ok) { reject(new Error(res?.error || 'Extension error')); return }
      resolve(res.payload ?? res)
    })
  })
}

export async function getCurrentTabId(): Promise<number | null> {
  if (!IS_EXTENSION) return null
  return new Promise(resolve => {
    (window as any).chrome.tabs.query({ active: true, currentWindow: true }, (tabs: any[]) =>
      resolve(tabs?.[0]?.id ?? null))
  })
}

export async function getTabState(tabId: number): Promise<AnalysisState> {
  if (!IS_EXTENSION) return MOCK_STATE
  return sendMsg({ type: 'GET_TAB_ANALYSIS', tabId })
}

export async function triggerAnalysis(tabId: number) {
  if (!IS_EXTENSION) return
  await sendMsg({ type: 'TRIGGER_ANALYSIS', tabId })
}

export async function askChat(tabId: number, question: string) {
  if (!IS_EXTENSION) return
  await sendMsg({ type: 'ASK_PAPER_CHAT', tabId, question })
}

export function onStateUpdate(tabId: number, cb: (s: AnalysisState) => void): () => void {
  if (!IS_EXTENSION) return () => {}
  const chrome = (window as any).chrome
  const fn = (msg: any) => { if (msg?.type === 'ANALYSIS_UPDATE' && msg.tabId === tabId) cb(msg.payload) }
  chrome.runtime.onMessage.addListener(fn)
  return () => chrome.runtime.onMessage.removeListener(fn)
}

export async function loadConfig(): Promise<ExtConfig> {
  if (!IS_EXTENSION) return {
    backendApiBase: 'http://localhost:8000', backendApiKey: '',
    llmApiKey: '', llmProvider: 'huggingface', userId: 'local-user',
  }
  const s = await (window as any).chrome.storage.local.get([
    'backendApiBase', 'backendApiKey', 'llmApiKey', 'llmProvider', 'openaiApiKey', 'userId',
  ])
  return {
    backendApiBase: s.backendApiBase || 'http://localhost:8000',
    backendApiKey: s.backendApiKey || '',
    llmApiKey: s.llmApiKey || s.openaiApiKey || '',
    llmProvider: (s.llmProvider as LLMProvider) || 'huggingface',
    userId: s.userId || 'local-user',
  }
}

export async function saveConfig(cfg: ExtConfig) {
  if (!IS_EXTENSION) return
  await (window as any).chrome.storage.local.set(cfg)
}
