import './App.css'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { PanelLeftClose, PanelLeftOpen, PanelRightClose, PanelRightOpen } from 'lucide-react'
import { saveSessionToFirebase } from './firebase.js'
import {
  ParticipantView,
  ParticipantsAudio,
  StreamCall,
  StreamVideo,
  StreamVideoClient,
  useCall,
  useCallStateHooks,
} from '@stream-io/video-react-sdk'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'
/** Gemini Live / native realtime default when switching provider back from OpenRouter VLM. */
const GEMINI_DEFAULT_MODEL = 'gemini-3.1-flash-live-preview'
/** Default OpenRouter VLM model (chat completions, not Live). */
const DEFAULT_OPENROUTER_VLM_MODEL = 'google/gemini-3.1-pro-preview'

/** OpenRouter VLM models shown in the model dropdown (provider = openrouter). */
const OPENROUTER_VLM_MODEL_OPTIONS = [
  { value: 'google/gemini-3.1-pro-preview', label: 'Gemini 3.1 Pro' },
  { value: 'google/gemini-3-flash-preview', label: 'Gemini 3 Flash' },
  { value: 'meta-llama/llama-4-scout', label: 'Llama 4 Scout' },
  { value: 'qwen/qwen3-vl-8b-instruct', label: 'Qwen3 VL 8B Instruct' },
  { value: 'anthropic/claude-sonnet-4.6', label: 'Claude Sonnet 4.6' },
  { value: 'openai/gpt-5.4', label: 'GPT-5.4' },
]

const OPENROUTER_VLM_MODEL_IDS = new Set(OPENROUTER_VLM_MODEL_OPTIONS.map((o) => o.value))

function ModelField({ provider, model, onModelChange }) {
  if (provider === 'openrouter') {
    const showCustom = model && !OPENROUTER_VLM_MODEL_IDS.has(model)
    return (
      <label>
        Model
        <select value={model} onChange={(e) => onModelChange(e.target.value)}>
          {OPENROUTER_VLM_MODEL_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
          {showCustom ? (
            <option value={model}>{model} (saved)</option>
          ) : null}
        </select>
      </label>
    )
  }
  return (
    <label>
      Model
      <input value={model} onChange={(e) => onModelChange(e.target.value)} />
    </label>
  )
}
const VLM_SYSTEM_INSTRUCTION_TEMPLATE = `You are an expert coach guiding a learner in a live session using {{SOFTWARE_TOOL}}.

This is real-time software coaching. You are looking at the learner's recent screen and your recent conversation with them.
Ground your response in what the learner is currently trying to do and what is visible from the immediate context.
Your response should sound like something a human coach would actually say aloud in real-time during the session.
Your goal is not only to complete the current tutorial, but to help the learner develop skill so that they can solve similar problems on their own later.

## Task
You are working on {{TASK_NAME}}.
{{TASK_DESCRIPTION}}

## Response Rules
- Be concise and to the point. Do not be overly verbose.
- Treat the prior transcript as a live conversation (Student = user, Coach = assistant).
- Be proactive when useful. If the learner seems stuck or pauses, offer a brief next step without waiting for a direct question.
- Avoid lists, overly formal language.
- Avoid any special characters or markdown formatting.
- Avoid transcript artifacts such as ellipses, dashes.
- Do not mention being an AI, model, assistant, or system.
- Do not describe hidden reasoning.

## Output rules
- Return ONLY the coach utterance as plain text.
- No JSON, no speaker prefix, no markdown.

## Few-shot examples

### Example 1: Direct Instruction
Software: OnShape
Task: Create a feature-based 3D part using sketches and extrusion.
Recent screen context: The learner has opened a new Part Studio. The Sketch toolbar is visible, and the Top plane is highlighted.
Recent transcript:
Student: I think I need to start drawing the base, but I'm not sure where to begin.
Coach: Start by making a sketch on one of the main planes.
Student: Okay, I clicked Sketch. Now what?
Correct output:
Select the Top plane, then choose the center-point rectangle tool.

### Example 2: Plan
Software: Blender
Task: Model a donut-like object with a torus, icing, and simple decoration.
Recent screen context: The learner has created the torus, but the icing mesh and sprinkles are not started yet.
Recent transcript:
Student: There are so many details in the tutorial. I don't know what to focus on first.
Coach: That's okay, we can break it down.
Student: Should I start making the sprinkles now?
Correct output:
Let’s get the main donut and icing shapes working first, and then we can add the small decorations.

### Example 3: Hint
Software: Figma
Task: Create a responsive card using Auto Layout and constraints.
Recent screen context: The learner has separate album art, title text, and metadata text. They are manually dragging the text when resizing the card.
Recent transcript:
Student: When I make the card wider, the text doesn't stay lined up with the image.
Coach: Right now the objects are still behaving like separate layers.
Student: So do I just keep moving them manually?
Correct output:
You probably want a layout feature that keeps those elements grouped and spaced automatically.

### Example 4: Diagnosis
Software: OnShape
Task: Create a solid object from a 2D sketch.
Recent screen context: The learner is trying to extrude a sketch, but the sketch profile is not selectable as a closed region.
Recent transcript:
Student: I clicked Extrude, but it won't let me select the inside shape.
Coach: Try clicking directly inside the profile.
Student: It still doesn't select anything.
Correct output:
The sketch looks like it has a small gap, so OnShape doesn't recognize it as a closed face yet.

### Example 5: Explanation
Software: Figma
Task: Create a responsive card component with Auto Layout.
Recent screen context: The learner has selected a frame containing album art and two text layers. The Auto Layout panel is open.
Recent transcript:
Student: Why do I need Auto Layout instead of just grouping everything?
Coach: Good question.
Correct output:
Auto Layout helps because it preserves spacing and alignment when the card content or size changes.

### Example 6: Tip
Software: Figma
Task: Create a reusable responsive card component.
Recent screen context: The learner has finished one card design and is about to duplicate it for variants.
Recent transcript:
Student: Should I just copy and paste this card a few times?
Coach: You can, but think about whether you'll need to reuse it.
Student: Yeah, probably with different episode titles.
Correct output:
A good habit is to turn the finished card into a component before making variants, so changes stay easier to manage later.

### Example 7: Clarification
Software: FL Studio
Task: Build a basic drum pattern and melody.
Recent screen context: The learner has several channels open in the channel rack and appears to be clicking between instruments.
Recent transcript:
Student: I don't know, this sounds kind of wrong.
Coach: I see.
Correct output:
Which part feels wrong to you right now, the rhythm or the instrument sound?

### Example 8: Reflection
Software: Figma
Task: Create an interactive loading animation.
Recent screen context: The learner previewed the prototype, but the animation jumps abruptly instead of smoothly transitioning.
Recent transcript:
Student: Huh, it just jumps to the next frame.
Coach: I saw that too.
Correct output:
Looking at that preview, what do you think is preventing it from animating smoothly?

### Example 9: Articulation
Software: OnShape
Task: Use feature-based modeling to create a beginner CAD object.
Recent screen context: The learner is hovering between Extrude, Revolve, and Fillet.
Recent transcript:
Student: I think maybe this one?
Coach: Before you click it, let's reason through it.
Correct output:
What are you expecting that tool to do to your sketch?

### Example 10: Exploration
Software: Blender
Task: Adjust material and appearance of a simple 3D object.
Recent screen context: The learner has the material properties panel open and is changing color and roughness values.
Recent transcript:
Student: Is this color okay?
Coach: It depends on the look you want.
Student: I want it to feel more like icing.
Correct output:
Try a couple of color and roughness settings and see which one feels most like icing to you.

### Example 11: Confirmation
Software: Figma
Task: Create a responsive card using Auto Layout and constraints.
Recent screen context: The learner resized the card, and the album art and text remain aligned with consistent spacing.
Recent transcript:
Student: Wait, now it actually stays aligned when I resize it.
Coach:
Correct output:
Yes, that's right. The Auto Layout is now controlling the spacing for you.

### Example 12: Non-Instructive
Software: Any software
Task: Any task.
Recent screen context: The learner is adjusting their microphone and the software task is not currently visible.
Recent transcript:
Student: Sorry, can you hear me now?
Coach:
Correct output:
Yes, I can hear you now.
`

const DEFAULT_TASK_DESCRIPTION =
  'Greet the user, confirm you can observe their screen share, and start coaching.'
const TASK_MARKDOWN_MODULES = import.meta.glob('./tasks/**/task.md', {
  eager: true,
  query: '?raw',
  import: 'default',
})

const extractTaskSection = (markdown, sectionTitle) => {
  const escapedTitle = sectionTitle.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const regex = new RegExp(
    `^#{2,6}\\s+${escapedTitle}\\s*\\n([\\s\\S]*?)(?=\\n#{1,6}\\s+|$)`,
    'im',
  )
  const match = markdown.match(regex)
  return match?.[1]?.trim() ?? ''
}

const taskFolderIdFromGlobPath = (path) => {
  const normalized = path.replace(/^\.\//, '')
  const parts = normalized.split('/').filter(Boolean)
  const file = parts[parts.length - 1]
  if (file !== 'task.md' || parts.length < 2) return null
  return parts[parts.length - 2] ?? null
}

const toTaskLabelFromPath = (path) => {
  const folderId = taskFolderIdFromGlobPath(path)
  if (folderId) return folderId
  return (
    path
      .split('/')
      .pop()
      ?.replace(/\.md$/i, '')
      .split(/[-_]/)
      .filter(Boolean)
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(' ') ?? 'Untitled Task'
  )
}

const buildTaskDescriptionFromMarkdown = (markdown) => {
  const taskDescription = extractTaskSection(markdown, 'Task Description')
  const learningObjectives = extractTaskSection(markdown, 'Learning Objectives')
  const intermediateSteps = extractTaskSection(markdown, 'Intermediate Steps')
  const finalOutcome = extractTaskSection(markdown, 'Final Outcome')

  const sections = [
    taskDescription ? `Task Description:\n${taskDescription}` : '',
    learningObjectives ? `Learning Objectives:\n${learningObjectives}` : '',
    intermediateSteps ? `Intermediate Steps:\n${intermediateSteps}` : '',
    finalOutcome ? `Final Outcome:\n${finalOutcome}` : '',
  ].filter(Boolean)

  return sections.join('\n\n') || DEFAULT_TASK_DESCRIPTION
}

const SOFTWARE_BY_TASK_PREFIX = {
  '01': 'Figma',
  '02': 'Blender',
  '03': 'Onshape',
  '04': 'FL Studio',
  '05': 'Excel',
}

function inferSoftwareTool(taskId) {
  return SOFTWARE_BY_TASK_PREFIX[taskId?.slice(0, 2)] ?? 'Figma'
}

function buildInstructionsForTask(task) {
  if (!task) return VLM_SYSTEM_INSTRUCTION_TEMPLATE
  return VLM_SYSTEM_INSTRUCTION_TEMPLATE.replace('{{SOFTWARE_TOOL}}', inferSoftwareTool(task.id))
    .replace('{{TASK_NAME}}', task.label)
    .replace('{{TASK_DESCRIPTION}}', task.description)
}

function taskConfigFromSelection(task) {
  return {
    task_id: task.id,
    taskDescription: task.description,
    instructions: buildInstructionsForTask(task),
  }
}

const TASK_OPTIONS = (() => {
  const entries = Object.entries(TASK_MARKDOWN_MODULES)
    .map(([path, markdown]) => {
      const id = taskFolderIdFromGlobPath(path) ?? crypto.randomUUID()
      const taskName = extractTaskSection(markdown, 'Task Name') || toTaskLabelFromPath(path)
      const description = buildTaskDescriptionFromMarkdown(markdown)
      return {
        id,
        label: taskName,
        description,
      }
    })
    .sort((a, b) => a.id.localeCompare(b.id, undefined, { numeric: true }))

  if (entries.length > 0) return entries

  return [
    {
      id: 'default',
      label: 'Default coaching',
      description: DEFAULT_TASK_DESCRIPTION,
    },
  ]
})()

function TaskContextCard({ taskId, tasksCatalog, fallbackTask }) {
  const fromCatalog = taskId && tasksCatalog ? tasksCatalog[taskId] : null
  const title = fromCatalog?.name?.trim() || fallbackTask?.label || ''
  const description = (fromCatalog?.description ?? fallbackTask?.description ?? '').trim()
  const link = (fromCatalog?.link ?? '').trim()

  if (tasksCatalog === null) {
    return (
      <div className="task-context-card" aria-busy="true">
        <p className="task-context-muted">Loading task details…</p>
      </div>
    )
  }

  if (!taskId || (!fromCatalog && !fallbackTask)) {
    return (
      <div className="task-context-card">
        <p className="task-context-muted">Select a task to see its description and document link.</p>
      </div>
    )
  }

  return (
    <div className="task-context-card">
      <h3 className="task-context-section-title">Task context</h3>
      {title ? <p className="task-context-task-title">{title}</p> : null}
      {description ? (
        <p className="task-context-description">
          <span className="task-context-description-label">Description:</span>{' '}
          <span className="task-context-description-body">{description}</span>
        </p>
      ) : (
        <p className="task-context-muted">No description available for this task.</p>
      )}
      {link ? (
        <a className="task-context-doc-link" href={link} target="_blank" rel="noopener noreferrer">
          Open task document
        </a>
      ) : null}
    </div>
  )
}

const normalizeTranscriptEntry = (entry) => {
  const timestamp = entry?.timestamp ?? ''
  const kind = entry?.kind ?? 'event'
  const text = entry?.text ?? ''
  const speaker = entry?.speaker ?? null
  const role = entry?.role ?? normalizeSpeakerRole(speaker)
  return { timestamp, kind, text, speaker, role }
}

function chatEntriesOnly(entries) {
  return entries.filter((e) => e?.kind === 'message' && String(e?.text ?? '').trim())
}

function chatBubbleLabel(role, speaker) {
  if (role === 'coach') return 'Agent'
  if (role === 'learner') return 'Learner'
  return (speaker ?? 'Participant').toString().trim() || 'Participant'
}

function ChatBubble({ entry }) {
  const normalized = normalizeTranscriptEntry(entry)
  if (normalized.kind !== 'message' || !String(normalized.text ?? '').trim()) {
    return null
  }
  const label = chatBubbleLabel(normalized.role, normalized.speaker)
  const side =
    normalized.role === 'coach' ? 'agent' : normalized.role === 'learner' ? 'learner' : 'other'
  return (
    <div className={`chat-bubble chat-bubble--${side}`}>
      <div className="chat-bubble-head">
        <span className="chat-bubble-label">{label}</span>
        <span className="chat-bubble-time">{normalized.timestamp}</span>
      </div>
      <div className="chat-bubble-body multiline-row">{normalized.text}</div>
    </div>
  )
}

function resolveStreamSpeakerRole(streamUser, callSession) {
  const uid = streamUser?.id != null ? String(streamUser.id) : ''
  const name = (streamUser?.name ?? '').toString().trim().toLowerCase()
  const localId = callSession?.user?.id != null ? String(callSession.user.id) : ''
  if (localId && uid === localId) {
    return {
      role: 'learner',
      speaker: streamUser?.name || callSession?.user?.name || 'Learner',
    }
  }
  if (uid === 'agent' || name.includes('digital coach') || name === 'coach' || name === 'assistant') {
    return { role: 'coach', speaker: streamUser?.name || 'Digital Coach' }
  }
  return {
    role: 'participant',
    speaker: streamUser?.name || streamUser?.id || 'Participant',
  }
}

const AGENT_TRANSCRIPT_PATTERN = /(?:🎤\s*)?\[\s*Agent transcript\s*\]\s*:\s*(.+)$/i
const USER_TRANSCRIPT_PATTERN = /(?:🎤\s*)?\[\s*User transcript\s*\]\s*:\s*(.+)$/i

function normalizeSpeakerRole(speaker) {
  const normalizedSpeaker = (speaker ?? '').toString().trim().toLowerCase()
  if (['coach', 'assistant', 'agent', 'digital coach', 'screen coach'].includes(normalizedSpeaker)) {
    return 'coach'
  }
  if (['learner', 'student', 'user', 'participant'].includes(normalizedSpeaker)) {
    return 'learner'
  }
  return 'participant'
}

const parseEmbeddedTranscriptEvent = (entry) => {
  const text = entry?.text ?? ''
  if (!text) return null
  const userMatch = text.match(USER_TRANSCRIPT_PATTERN)
  if (userMatch?.[1]) {
    return {
      ...entry,
      kind: 'message',
      speaker: 'Learner',
      role: 'learner',
      text: userMatch[1].trim(),
    }
  }
  const agentMatch = text.match(AGENT_TRANSCRIPT_PATTERN)
  if (agentMatch?.[1]) {
    return {
      ...entry,
      kind: 'message',
      speaker: 'Digital Coach',
      role: 'coach',
      text: agentMatch[1].trim(),
    }
  }
  return null
}

function CallContent({
  callSession,
  onLeave,
  onStart,
  isStarting,
  isConnected,
  statusMessage,
  setStatusMessage,
  modelConfig,
  setModelConfig,
  saveModelConfig,
  transcriptEntries,
  appendTranscriptEntry,
  isTranscriptVisible,
  onToggleTranscript,
  selectedTaskId,
  onTaskChange,
  tasksCatalog,
  isStatsVisible,
  onToggleStats,
}) {
  const call = useCall()
  const { useParticipants, useIsAutoplayBlocked } = useCallStateHooks()
  const participants = useParticipants()
  const isAutoplayBlocked = useIsAutoplayBlocked()
  const [collapsedSections, setCollapsedSections] = useState({
    agentConfig: true,
    task: false,
    callStatus: false,
    callId: false,
    model: false,
    prompts: false,
    performance: false,
  })
  const [isMicOn, setIsMicOn] = useState(false)
  const [isCameraOn, setIsCameraOn] = useState(false)
  const [isSharing, setIsSharing] = useState(false)
  const chatMessages = useMemo(() => chatEntriesOnly(transcriptEntries), [transcriptEntries])
  useEffect(() => {
    if (!call) return undefined

    call.startClosedCaptions().catch((err) => {
      const message = String(err?.message || err || '')
      if (message.includes('already being transcribed')) return
      setStatusMessage(`Captions unavailable: ${message || 'Unknown error'}`)
    })

    return undefined
  }, [call, setStatusMessage])

  useEffect(() => {
    if (!call || !callSession) return
    setStatusMessage(`Connected (${callSession.callId})`)
    call.microphone
      .enable()
      .then(() => {
        setIsMicOn(true)
        setStatusMessage('Connected. Mic enabled.')
      })
      .catch((err) => {
        if (err?.name === 'NotAllowedError') {
          setStatusMessage('Mic blocked. Allow microphone permission to talk with the realtime agent.')
          return
        }
        setStatusMessage(`Mic setup failed: ${err?.message || 'Unknown error'}`)
      })
  }, [call, callSession, setStatusMessage])

  useEffect(() => {
    if (!call) return undefined
    const unsubscribes = [
      call.on('call.closed_caption', (event) => {
        const text = event?.caption || event?.text || ''
        if (!text.trim()) return
        const { role, speaker } = resolveStreamSpeakerRole(event?.user, callSession)
        appendTranscriptEntry({
          kind: 'message',
          speaker,
          role,
          text,
        })
      }),
      call.on('call.message_new', (event) => {
        const text = event?.message?.text || ''
        if (!text.trim()) return
        const { role, speaker } = resolveStreamSpeakerRole(event?.message?.user, callSession)
        appendTranscriptEntry({
          kind: 'message',
          speaker,
          role,
          text,
        })
      }),
    ]
    return () => {
      unsubscribes.forEach((unsubscribe) => {
        if (typeof unsubscribe === 'function') unsubscribe()
      })
    }
  }, [appendTranscriptEntry, call, callSession])

  const remoteParticipant = useMemo(
    () => participants.find((p) => p.userId !== callSession?.user?.id) ?? null,
    [participants, callSession?.user?.id],
  )
  const localParticipant = useMemo(
    () => participants.find((p) => p.userId === callSession?.user?.id) ?? null,
    [participants, callSession?.user?.id],
  )
  const screenShareParticipant = useMemo(() => {
    const hasScreenShare = (participant) =>
      Boolean(
        participant?.screenShareTrack ||
          participant?.screenShareStream ||
          participant?.publishedTracks?.screenShareTrack ||
          participant?.isScreenSharing ||
          participant?.isScreensharing,
      )

    const localScreenShare = participants.find(
      (participant) => participant?.userId === callSession?.user?.id && hasScreenShare(participant),
    )
    if (localScreenShare) return localScreenShare

    return participants.find((participant) => hasScreenShare(participant)) ?? null
  }, [callSession?.user?.id, participants])
  const agentConfig = callSession?.agentConfig ?? {}
  const systemPrompt =
    agentConfig.systemPrompt ?? agentConfig.system_prompt ?? agentConfig.instructions ?? 'Not available'
  const taskDescription =
    agentConfig.task_description ??
    agentConfig.taskDescription ??
    DEFAULT_TASK_DESCRIPTION
  const taskName = agentConfig.task_name ?? agentConfig.taskName ?? null
  const contextWindow =
    agentConfig.contextWindow ?? agentConfig.context_window ?? agentConfig.context_length ?? null
  const fps = agentConfig.fps ?? null

  const toggleSection = (key) => {
    setCollapsedSections((prev) => ({ ...prev, [key]: !prev[key] }))
  }

  const handleShare = async () => {
    if (!call) return
    try {
      if (isSharing) {
        await call.screenShare.disable()
        setIsSharing(false)
        setStatusMessage('Screen share stopped')
        return
      }
      await call.screenShare.enable()
      setIsSharing(true)
      setStatusMessage('Screen share started')
    } catch (err) {
      if (err?.name === 'NotAllowedError') {
        setStatusMessage('Screen share blocked. Allow screen recording permission in browser/system settings.')
        return
      }
      setStatusMessage(`Screen share failed: ${err?.message || 'Unknown error'}`)
    }
  }

  const handleMicToggle = async () => {
    if (!call) return
    try {
      if (isMicOn) {
        await call.microphone.disable()
        setIsMicOn(false)
        setStatusMessage('Mic muted')
        return
      }
      await call.microphone.enable()
      try {
        await call.resumeAudio()
      } catch {
        /* browser may still require explicit “enable sound” if autoplay blocked */
      }
      setIsMicOn(true)
      setStatusMessage('Mic unmuted')
    } catch (err) {
      if (err?.name === 'NotAllowedError') {
        setStatusMessage('Mic blocked. Allow microphone permission to talk with the realtime agent.')
        return
      }
      setStatusMessage(`Mic control failed: ${err?.message || 'Unknown error'}`)
    }
  }

  const handleCameraToggle = async () => {
    if (!call) return
    try {
      if (isCameraOn) {
        await call.camera.disable()
        setIsCameraOn(false)
        setStatusMessage('Camera disabled')
        return
      }
      await call.camera.enable()
      setIsCameraOn(true)
      setStatusMessage('Camera enabled')
    } catch (err) {
      if (err?.name === 'NotAllowedError') {
        setStatusMessage('Camera blocked. Allow camera permission to enable video.')
        return
      }
      setStatusMessage(`Camera control failed: ${err?.message || 'Unknown error'}`)
    }
  }

  const handleResumeCoachAudio = async () => {
    if (!call) return
    try {
      await call.resumeAudio()
      setStatusMessage('Coach audio enabled')
    } catch (err) {
      setStatusMessage(`Could not start coach audio: ${err?.message || 'Unknown error'}`)
    }
  }

  return (
    <div
      className={`call-ui ${isTranscriptVisible ? '' : 'transcript-hidden'} ${isStatsVisible ? '' : 'stats-hidden'}`}
    >
      <ParticipantsAudio participants={participants} />
      {isAutoplayBlocked ? (
        <div className="autoplay-audio-banner" role="status">
          <p>The browser blocked automatic coach audio.</p>
          <button type="button" className="autoplay-audio-banner__action" onClick={handleResumeCoachAudio}>
            Enable coach audio
          </button>
        </div>
      ) : null}
      {isStatsVisible ? (
        <aside className="panel panel-left">
        <div className="panel-header panel-header-leading">
          <button className="icon-button" type="button" aria-label="Hide statistics" onClick={onToggleStats}>
            <PanelLeftClose size={16} />
          </button>
          <h2>Statistics</h2>
        </div>
        <div className="section-list">
          <section>
            <button className="section-title" type="button" onClick={() => toggleSection('agentConfig')}>
              <span>Agent configuration</span>
              <span>{collapsedSections.agentConfig ? '›' : '⌄'}</span>
            </button>
            {!collapsedSections.agentConfig ? (
              <div className="config-card">
                <label>
                  Provider
                  <select
                    value={modelConfig.provider}
                    onChange={(e) => {
                      const p = e.target.value
                      setModelConfig((prev) => {
                        if (p === 'openrouter') {
                          return { ...prev, provider: p, model: DEFAULT_OPENROUTER_VLM_MODEL }
                        }
                        if (prev.provider === 'openrouter') {
                          const nextModel = OPENROUTER_VLM_MODEL_IDS.has(prev.model)
                            ? GEMINI_DEFAULT_MODEL
                            : prev.model
                          return { ...prev, provider: p, model: nextModel }
                        }
                        return { ...prev, provider: p }
                      })
                    }}
                  >
                    <option value="gemini">gemini</option>
                    <option value="openai">openai</option>
                    <option value="qwen">qwen</option>
                    <option value="openrouter">openrouter (VLM)</option>
                  </select>
                </label>
                <ModelField
                  provider={modelConfig.provider}
                  model={modelConfig.model}
                  onModelChange={(model) => setModelConfig((prev) => ({ ...prev, model }))}
                />
                <label>
                  FPS
                  <input
                    type="number"
                    min="1"
                    max="15"
                    value={modelConfig.fps}
                    onChange={(e) =>
                      setModelConfig((prev) => ({ ...prev, fps: Number(e.target.value) || 1 }))
                    }
                  />
                </label>
                {modelConfig.provider === 'openrouter' ? (
                  <label>
                    Frame buffer (seconds, VLM)
                    <input
                      type="number"
                      min="1"
                      max="120"
                      value={modelConfig.frame_buffer_seconds}
                      onChange={(e) =>
                        setModelConfig((prev) => ({
                          ...prev,
                          frame_buffer_seconds: Number(e.target.value) || 1,
                        }))
                      }
                    />
                  </label>
                ) : null}
                <label>
                  Proactive coaching
                  <select
                    value={modelConfig.proactive_enabled ? 'enabled' : 'disabled'}
                    onChange={(e) =>
                      setModelConfig((prev) => ({
                        ...prev,
                        proactive_enabled: e.target.value === 'enabled',
                      }))
                    }
                  >
                    <option value="enabled">enabled</option>
                    <option value="disabled">disabled</option>
                  </select>
                </label>
                <label>
                  Fire rate (seconds)
                  <input
                    type="number"
                    min="5"
                    max="300"
                    value={modelConfig.proactive_fire_rate_seconds}
                    onChange={(e) =>
                      setModelConfig((prev) => ({
                        ...prev,
                        proactive_fire_rate_seconds: Number(e.target.value) || 30,
                      }))
                    }
                  />
                </label>
                <label>
                  Instructions
                  <textarea
                    value={modelConfig.instructions}
                    onChange={(e) =>
                      setModelConfig((prev) => ({ ...prev, instructions: e.target.value }))
                    }
                  />
                </label>
                <button type="button" onClick={saveModelConfig} className="control-chip">
                  Save Agent Config
                </button>
              </div>
            ) : null}
          </section>
          <section>
            <button className="section-title" type="button" onClick={() => toggleSection('task')}>
              <span>Task</span>
              <span>{collapsedSections.task ? '›' : '⌄'}</span>
            </button>
            {!collapsedSections.task ? (
              <>
                <select className="task-select" value={selectedTaskId} onChange={(e) => onTaskChange(e.target.value)}>
                  {TASK_OPTIONS.map((task) => (
                    <option key={task.id} value={task.id}>
                      {task.label}
                    </option>
                  ))}
                </select>
                <TaskContextCard
                  taskId={selectedTaskId}
                  tasksCatalog={tasksCatalog}
                  fallbackTask={TASK_OPTIONS.find((t) => t.id === selectedTaskId)}
                />
              </>
            ) : null}
          </section>
          <section>
            <button className="section-title" type="button" onClick={() => toggleSection('callStatus')}>
              <span>Call status</span>
              <span>{collapsedSections.callStatus ? '›' : '⌄'}</span>
            </button>
            {!collapsedSections.callStatus ? <p className="loading-row">{statusMessage}</p> : null}
          </section>
          <section>
            <button className="section-title" type="button" onClick={() => toggleSection('callId')}>
              <span>Call id</span>
              <span>{collapsedSections.callId ? '›' : '⌄'}</span>
            </button>
            {!collapsedSections.callId ? <p className="loading-row">{callSession.callId}</p> : null}
          </section>
          <section>
            <button className="section-title" type="button" onClick={() => toggleSection('model')}>
              <span>Model</span>
              <span>{collapsedSections.model ? '›' : '⌄'}</span>
            </button>
            {!collapsedSections.model ? (
              <>
                <p className="loading-row">{agentConfig.provider ?? 'Not available'}</p>
                <p className="loading-row">{agentConfig.model ?? 'Not available'}</p>
              </>
            ) : null}
          </section>
          <section>
            <button className="section-title" type="button" onClick={() => toggleSection('prompts')}>
              <span>Prompts</span>
              <span>{collapsedSections.prompts ? '›' : '⌄'}</span>
            </button>
            {!collapsedSections.prompts ? (
              <>
                <p className="loading-row label-row">System Prompt</p>
                <p className="loading-row multiline-row">{systemPrompt}</p>
                {taskName ? (
                  <>
                    <p className="loading-row label-row">Task</p>
                    <p className="loading-row">{taskName}</p>
                  </>
                ) : null}
                <p className="loading-row label-row">Task Description</p>
                <p className="loading-row multiline-row">{taskDescription}</p>
              </>
            ) : null}
          </section>
          <section>
            <button className="section-title" type="button" onClick={() => toggleSection('performance')}>
              <span>Performance</span>
              <span>{collapsedSections.performance ? '›' : '⌄'}</span>
            </button>
            {!collapsedSections.performance ? (
              <>
                <p className="loading-row label-row">Context Window</p>
                <p className="loading-row">
                  {contextWindow !== null && contextWindow !== undefined
                    ? String(contextWindow)
                    : 'Not available'}
                </p>
                <p className="loading-row label-row">FPS</p>
                <p className="loading-row">
                  {fps !== null && fps !== undefined ? String(fps) : 'Not available'}
                </p>
              </>
            ) : null}
          </section>
        </div>
        </aside>
      ) : null}

      <main className="panel panel-main">
        {!isStatsVisible ? (
          <button className="stats-reveal-button" type="button" aria-label="Show statistics" onClick={onToggleStats}>
            <PanelLeftOpen size={16} />
            Show statistics
          </button>
        ) : null}
        {!isTranscriptVisible ? (
          <button
            className="transcript-reveal-button"
            type="button"
            aria-label="Show chat"
            onClick={onToggleTranscript}
          >
            <PanelRightOpen size={16} />
            Show Chat
          </button>
        ) : null}
        <div
          className={`call-stage ${isConnected && (screenShareParticipant || remoteParticipant) ? '' : 'call-stage-centered'}`}
        >
          {isConnected && screenShareParticipant ? (
            <div className="participant-stage participant-stage-screen-share">
              <ParticipantView participant={screenShareParticipant} trackType="screenShareTrack" />
            </div>
          ) : isConnected && remoteParticipant ? (
            <div className="participant-stage participant-stage-camera">
              <ParticipantView participant={remoteParticipant} />
            </div>
          ) : (
            <div className="stage-placeholder">
              <p className="loading-row">
                {isConnected ? 'Share your screen or wait for Digital Coach stream...' : 'Ready to start your session'}
              </p>
            </div>
          )}
        </div>
        <footer className="call-controls">
          <button className={`control-chip ${isMicOn ? '' : 'muted'}`} type="button" onClick={handleMicToggle}>
            <span className="chip-icon">🎙</span>
            {isMicOn ? 'Mic On' : 'Mic Off'}
          </button>
          <button
            className={`control-chip ${isCameraOn ? '' : 'muted'}`}
            type="button"
            onClick={handleCameraToggle}
          >
            <span className="chip-icon">📷</span>
            {isCameraOn ? 'Camera On' : 'Camera Off'}
          </button>
          <button className="share-button" type="button" onClick={handleShare} disabled={!isConnected || isSharing}>
            {isSharing ? 'Sharing' : 'Share'}
          </button>
          <button
            className={isConnected ? 'leave-button' : 'share-button'}
            type="button"
            onClick={isConnected ? onLeave : onStart}
            disabled={isStarting}
          >
            {isConnected ? 'End Session' : isStarting ? 'Starting...' : 'Start Session'}
          </button>
        </footer>
      </main>

      <aside className="panel panel-right">
        <div className="panel-header panel-header-leading">
          <button className="icon-button" type="button" aria-label="Hide chat" onClick={onToggleTranscript}>
            <PanelRightClose size={16} />
          </button>
          <h2>Chat</h2>
        </div>
        {isTranscriptVisible ? (
          <div className="transcript-body chat-thread">
            {chatMessages.length ? (
              chatMessages.map((entry) => <ChatBubble key={entry.id} entry={entry} />)
            ) : (
              <p className="chat-thread-empty">No messages yet.</p>
            )}
          </div>
        ) : null}
      </aside>
    </div>
  )
}

function PreSessionContent({
  onStart,
  isStarting,
  statusMessage,
  modelConfig,
  setModelConfig,
  saveModelConfig,
  transcriptEntries,
  isTranscriptVisible,
  onToggleTranscript,
  selectedTaskId,
  onTaskChange,
  tasksCatalog,
  isStatsVisible,
  onToggleStats,
}) {
  const [collapsedSections, setCollapsedSections] = useState({
    agentConfig: true,
    task: false,
    callStatus: false,
  })

  const toggleSection = (key) => {
    setCollapsedSections((prev) => ({ ...prev, [key]: !prev[key] }))
  }

  const chatMessages = useMemo(() => chatEntriesOnly(transcriptEntries), [transcriptEntries])

  return (
    <div
      className={`call-ui ${isTranscriptVisible ? '' : 'transcript-hidden'} ${isStatsVisible ? '' : 'stats-hidden'}`}
    >
      {isStatsVisible ? (
        <aside className="panel panel-left">
        <div className="panel-header panel-header-leading">
          <button className="icon-button" type="button" aria-label="Hide statistics" onClick={onToggleStats}>
            <PanelLeftClose size={16} />
          </button>
          <h2>Statistics</h2>
        </div>
        <div className="section-list">
          <section>
            <button className="section-title" type="button" onClick={() => toggleSection('agentConfig')}>
              <span>Agent configuration</span>
              <span>{collapsedSections.agentConfig ? '›' : '⌄'}</span>
            </button>
            {!collapsedSections.agentConfig ? (
              <div className="config-card">
                <label>
                  Provider
                  <select
                    value={modelConfig.provider}
                    onChange={(e) => {
                      const p = e.target.value
                      setModelConfig((prev) => {
                        if (p === 'openrouter') {
                          return { ...prev, provider: p, model: DEFAULT_OPENROUTER_VLM_MODEL }
                        }
                        if (prev.provider === 'openrouter') {
                          const nextModel = OPENROUTER_VLM_MODEL_IDS.has(prev.model)
                            ? GEMINI_DEFAULT_MODEL
                            : prev.model
                          return { ...prev, provider: p, model: nextModel }
                        }
                        return { ...prev, provider: p }
                      })
                    }}
                  >
                    <option value="gemini">gemini</option>
                    <option value="openai">openai</option>
                    <option value="qwen">qwen</option>
                    <option value="openrouter">openrouter (VLM)</option>
                  </select>
                </label>
                <ModelField
                  provider={modelConfig.provider}
                  model={modelConfig.model}
                  onModelChange={(model) => setModelConfig((prev) => ({ ...prev, model }))}
                />
                <label>
                  FPS
                  <input
                    type="number"
                    min="1"
                    max="15"
                    value={modelConfig.fps}
                    onChange={(e) => setModelConfig((prev) => ({ ...prev, fps: Number(e.target.value) || 1 }))}
                  />
                </label>
                {modelConfig.provider === 'openrouter' ? (
                  <label>
                    Frame buffer (seconds, VLM)
                    <input
                      type="number"
                      min="1"
                      max="120"
                      value={modelConfig.frame_buffer_seconds}
                      onChange={(e) =>
                        setModelConfig((prev) => ({
                          ...prev,
                          frame_buffer_seconds: Number(e.target.value) || 1,
                        }))
                      }
                    />
                  </label>
                ) : null}
                <label>
                  Proactive coaching
                  <select
                    value={modelConfig.proactive_enabled ? 'enabled' : 'disabled'}
                    onChange={(e) =>
                      setModelConfig((prev) => ({
                        ...prev,
                        proactive_enabled: e.target.value === 'enabled',
                      }))
                    }
                  >
                    <option value="enabled">enabled</option>
                    <option value="disabled">disabled</option>
                  </select>
                </label>
                <label>
                  Fire rate (seconds)
                  <input
                    type="number"
                    min="5"
                    max="300"
                    value={modelConfig.proactive_fire_rate_seconds}
                    onChange={(e) =>
                      setModelConfig((prev) => ({
                        ...prev,
                        proactive_fire_rate_seconds: Number(e.target.value) || 30,
                      }))
                    }
                  />
                </label>
                <label>
                  Instructions
                  <textarea
                    value={modelConfig.instructions}
                    onChange={(e) => setModelConfig((prev) => ({ ...prev, instructions: e.target.value }))}
                  />
                </label>
                <button type="button" onClick={saveModelConfig} className="control-chip">
                  Save Agent Config
                </button>
              </div>
            ) : null}
          </section>
          <section>
            <button className="section-title" type="button" onClick={() => toggleSection('task')}>
              <span>Task</span>
              <span>{collapsedSections.task ? '›' : '⌄'}</span>
            </button>
            {!collapsedSections.task ? (
              <>
                <select className="task-select" value={selectedTaskId} onChange={(e) => onTaskChange(e.target.value)}>
                  {TASK_OPTIONS.map((task) => (
                    <option key={task.id} value={task.id}>
                      {task.label}
                    </option>
                  ))}
                </select>
                <TaskContextCard
                  taskId={selectedTaskId}
                  tasksCatalog={tasksCatalog}
                  fallbackTask={TASK_OPTIONS.find((t) => t.id === selectedTaskId)}
                />
              </>
            ) : null}
          </section>
          <section>
            <button className="section-title" type="button" onClick={() => toggleSection('callStatus')}>
              <span>Call status</span>
              <span>{collapsedSections.callStatus ? '›' : '⌄'}</span>
            </button>
            {!collapsedSections.callStatus ? <p className="loading-row">{statusMessage}</p> : null}
          </section>
        </div>
        </aside>
      ) : null}

      <main className="panel panel-main">
        {!isStatsVisible ? (
          <button className="stats-reveal-button" type="button" aria-label="Show statistics" onClick={onToggleStats}>
            <PanelLeftOpen size={16} />
            Show statistics
          </button>
        ) : null}
        {!isTranscriptVisible ? (
          <button
            className="transcript-reveal-button"
            type="button"
            aria-label="Show chat"
            onClick={onToggleTranscript}
          >
            <PanelRightOpen size={16} />
            Show Chat
          </button>
        ) : null}
        <div className="call-stage call-stage-centered">
          <div className="stage-placeholder">
            <p className="loading-row">Ready to start your session</p>
          </div>
        </div>
        <footer className="call-controls">
          <button className="share-button" type="button" onClick={onStart} disabled={isStarting}>
            {isStarting ? 'Starting...' : 'Start Session'}
          </button>
        </footer>
      </main>

      <aside className="panel panel-right">
        <div className="panel-header panel-header-leading">
          <button className="icon-button" type="button" aria-label="Hide chat" onClick={onToggleTranscript}>
            <PanelRightClose size={16} />
          </button>
          <h2>Chat</h2>
        </div>
        {isTranscriptVisible ? (
          <div className="transcript-body chat-thread">
            {chatMessages.length ? (
              chatMessages.map((entry) => <ChatBubble key={entry.id} entry={entry} />)
            ) : (
              <p className="chat-thread-empty">No messages yet. Start a session to chat.</p>
            )}
          </div>
        ) : null}
      </aside>
    </div>
  )
}

const SOFTWARE_GROUPS = [
  { prefix: '01', label: 'Figma' },
  { prefix: '02', label: 'Blender' },
  { prefix: '03', label: 'Onshape' },
  { prefix: '04', label: 'FL Studio' },
  { prefix: '05', label: 'Excel' },
]

function LoginPage({ onLogin }) {
  const [userId, setUserId] = useState('')
  const [taskId, setTaskId] = useState(TASK_OPTIONS[0]?.id ?? '')
  const [error, setError] = useState('')
  const inputRef = useRef(null)

  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  const handleSubmit = (e) => {
    e.preventDefault()
    const trimmed = userId.trim()
    if (!trimmed) {
      setError('Please enter your ID to continue.')
      return
    }
    onLogin(trimmed, taskId)
  }

  const grouped = SOFTWARE_GROUPS.map((g) => ({
    ...g,
    tasks: TASK_OPTIONS.filter((t) => t.id.startsWith(g.prefix)),
  })).filter((g) => g.tasks.length > 0)

  return (
    <div className="login-page">
      <div className="login-card">
        <div className="login-logo">
          <span className="login-logo-icon">🏋️</span>
        </div>
        <h1 className="login-title">Digital Coach Gym</h1>
        <p className="login-subtitle">Enter your ID and choose a task to begin your session.</p>

        <form className="login-form" onSubmit={handleSubmit} noValidate>
          <div className="login-field">
            <label className="login-label" htmlFor="login-user-id">Your ID</label>
            <input
              ref={inputRef}
              id="login-user-id"
              className={`login-input${error ? ' login-input-error' : ''}`}
              type="text"
              placeholder="e.g. A1"
              value={userId}
              onChange={(e) => { setUserId(e.target.value); setError('') }}
              autoComplete="off"
              spellCheck={false}
            />
            {error ? <p className="login-error">{error}</p> : null}
          </div>

          <div className="login-field">
            <label className="login-label" htmlFor="login-task-select">Task</label>
            <select
              id="login-task-select"
              className="login-select"
              value={taskId}
              onChange={(e) => setTaskId(e.target.value)}
            >
              {grouped.map((group) => (
                <optgroup key={group.prefix} label={group.label}>
                  {group.tasks.map((task) => (
                    <option key={task.id} value={task.id}>
                      {task.id} — {task.label}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </div>

          <button className="login-button" type="submit">
            Enter
          </button>
        </form>
      </div>
    </div>
  )
}

function App() {
  const [currentUser, setCurrentUser] = useState(null)
  const [client, setClient] = useState(null)
  const [call, setCall] = useState(null)
  const [callSession, setCallSession] = useState(null)
  const [statusMessage, setStatusMessage] = useState('Ready')
  const [isStarting, setIsStarting] = useState(false)
  const [isTranscriptVisible, setIsTranscriptVisible] = useState(true)
  const [isStatsVisible, setIsStatsVisible] = useState(true)
  const [selectedTaskId, setSelectedTaskId] = useState(TASK_OPTIONS[0].id)
  const [tasksCatalog, setTasksCatalog] = useState(null)
  const [transcriptEntries, setTranscriptEntries] = useState([])
  const [modelConfig, setModelConfig] = useState(() => ({
    provider: 'openrouter',
    model: DEFAULT_OPENROUTER_VLM_MODEL,
    fps: 1,
    frame_buffer_seconds: 10,
    proactive_enabled: true,
    proactive_fire_rate_seconds: 30,
    ...taskConfigFromSelection(TASK_OPTIONS[0]),
  }))

  const handleLogin = useCallback((userId, taskId) => {
    setCurrentUser({ id: userId })
    const selectedTask = TASK_OPTIONS.find((t) => t.id === taskId) ?? TASK_OPTIONS[0]
    setSelectedTaskId(selectedTask.id)
    setModelConfig((prev) => ({ ...prev, ...taskConfigFromSelection(selectedTask) }))
  }, [])

  const createTranscriptEntry = useCallback((entry) => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`
    // Prefer a unix timestamp (ms) already on the entry; otherwise stamp now.
    const resolveTs = (existing) =>
      typeof existing === 'number' ? existing : Date.now()
    if (typeof entry === 'string') {
      const parsedStringEvent = parseEmbeddedTranscriptEvent({ text: entry })
      if (parsedStringEvent) {
        return {
          id,
          kind: 'message',
          text: parsedStringEvent.text,
          speaker: parsedStringEvent.speaker,
          role: parsedStringEvent.role,
          timestamp: Date.now(),
        }
      }
      return { id, kind: 'event', text: entry, timestamp: Date.now(), speaker: null }
    }
    // Backward compatibility for any legacy `{ id, text }` transcript rows.
    if (entry && typeof entry === 'object' && !('kind' in entry) && 'text' in entry) {
      const parsedLegacyEvent = parseEmbeddedTranscriptEvent(entry)
      if (parsedLegacyEvent) {
        return {
          id,
          kind: 'message',
          text: parsedLegacyEvent.text,
          speaker: parsedLegacyEvent.speaker,
          role: parsedLegacyEvent.role,
          timestamp: resolveTs(entry.timestamp),
        }
      }
      return {
        id,
        kind: 'event',
        text: entry.text ?? '',
        speaker: null,
        timestamp: resolveTs(entry.timestamp),
      }
    }
    const parsedEvent = parseEmbeddedTranscriptEvent(entry)
    if (parsedEvent) {
      return {
        id,
        kind: 'message',
        text: parsedEvent.text,
        speaker: parsedEvent.speaker,
        role: parsedEvent.role,
        timestamp: Date.now(),
      }
    }
    const speaker = entry?.speaker ?? null
    const inferredRole = entry?.role ?? (speaker ? normalizeSpeakerRole(speaker) : null)
    return {
      id,
      kind: entry?.kind ?? 'event',
      text: entry?.text ?? '',
      speaker,
      role: inferredRole,
      timestamp: resolveTs(entry?.timestamp),
    }
  }, [])

  useEffect(() => {
    fetch('/tasks.json')
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => setTasksCatalog(data && typeof data === 'object' ? data : {}))
      .catch(() => setTasksCatalog({}))
  }, [])

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/agent/config`)
      .then((res) => (res.ok ? res.json() : null))
      .then((cfg) => {
        if (!cfg) return
        const taskId = cfg.task_id ?? TASK_OPTIONS[0].id
        const selectedTask = TASK_OPTIONS.find((task) => task.id === taskId) ?? TASK_OPTIONS[0]
        setSelectedTaskId(selectedTask.id)
        setModelConfig((prev) => ({
          ...prev,
          ...cfg,
          instructions: cfg.instructions || buildInstructionsForTask(selectedTask),
          proactive_enabled: cfg.proactive_enabled ?? prev.proactive_enabled,
          proactive_fire_rate_seconds:
            cfg.proactive_fire_rate_seconds ?? prev.proactive_fire_rate_seconds,
          frame_buffer_seconds: cfg.frame_buffer_seconds ?? prev.frame_buffer_seconds,
          taskDescription: cfg.task_description ?? cfg.taskDescription ?? selectedTask.description,
          task_id: selectedTask.id,
        }))
      })
      .catch(() => {})
  }, [])

  const persistModelConfig = useCallback(async (config) => {
    const res = await fetch(`${API_BASE_URL}/api/agent/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    })
    if (!res.ok) throw new Error(await res.text())
    const data = await res.json()
    return data.config ?? config
  }, [])

  const startCall = async () => {
    setIsStarting(true)
    setStatusMessage('Creating session...')
    setTranscriptEntries([])
    try {
      setStatusMessage('Syncing task config...')
      const savedConfig = await persistModelConfig(modelConfig)
      setModelConfig((prev) => ({
        ...prev,
        ...savedConfig,
        taskDescription:
          savedConfig.task_description ?? savedConfig.taskDescription ?? prev.taskDescription,
      }))
      const res = await fetch(`${API_BASE_URL}/api/session/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: currentUser?.id ?? undefined,
          user_name: currentUser?.id ?? 'Learner',
          call_type: 'default',
          task_id: selectedTaskId,
        }),
      })
      if (!res.ok) {
        throw new Error(await res.text())
      }
      const session = await res.json()
      const nextClient = new StreamVideoClient({
        apiKey: session.apiKey,
        user: session.user,
        token: session.token,
      })
      const nextCall = nextClient.call(session.callType, session.callId)
      await nextCall.join({ create: true })
      try {
        await nextCall.resumeAudio()
      } catch {
        /* remote tracks may not exist yet; CallContent shows “Enable coach audio” if still blocked */
      }
      setClient(nextClient)
      setCall(nextCall)
      setCallSession(session)
      setStatusMessage('Connected')
    } catch (err) {
      setStatusMessage(`Connection failed: ${err.message}`)
    } finally {
      setIsStarting(false)
    }
  }

  const leaveCall = async () => {
    if (!call || !client || !callSession) return
    setStatusMessage('Leaving...')
    const savedCallId = callSession.callId
    const savedSessionId = callSession.sessionId
    const savedAt = Date.now()
    const chatOnly = transcriptEntries.filter((e) => (e.kind ?? 'event') === 'message')
    const entriesSnapshot = chatOnly.map((e) => ({
      id: e.id,
      kind: e.kind ?? 'message',
      text: e.text ?? '',
      speaker: e.speaker ?? null,
      role: e.role ?? null,
      timestamp: typeof e.timestamp === 'number' ? e.timestamp : savedAt,
    }))
    try {
      try {
        const saveRes = await fetch(`${API_BASE_URL}/api/session/${savedCallId}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: savedSessionId,
            task_id: selectedTaskId,
            user: callSession.user,
            user_id: currentUser?.id ?? null,
            saved_at: savedAt,
            entries: entriesSnapshot,
          }),
        })
        if (saveRes.ok) {
          const saved = await saveRes.json()
          setStatusMessage(`Chat saved (${saved.entry_count ?? entriesSnapshot.length} messages): ${saved.filename ?? 'export'}`)
        } else {
          const errText = await saveRes.text()
          setStatusMessage(`Chat save failed: ${saveRes.status} ${errText}`)
        }
      } catch (saveErr) {
        setStatusMessage(`Chat save failed: ${saveErr?.message || saveErr}`)
      }
      // Upload to Firebase (best-effort, non-blocking)
      saveSessionToFirebase({
        userId: currentUser?.id ?? callSession.user?.id ?? 'anonymous',
        sessionId: savedSessionId,
        callId: savedCallId,
        taskId: selectedTaskId,
        savedAt,
        entries: entriesSnapshot,
      }).catch((err) => console.warn('[Firebase] Upload failed:', err))
      // Leave the browser participant so SFU state is settled before
      // requesting backend agent session shutdown.
      await call.leave()
      await fetch(`${API_BASE_URL}/api/session/${savedCallId}/${savedSessionId}/close`, {
        method: 'POST',
      })
      await client.disconnectUser()
    } catch {
      // best effort cleanup
    } finally {
      // Keep transcriptEntries intact — live poll already populated it during the session.
      setCall(null)
      setClient(null)
      setCallSession(null)
      setStatusMessage('Ready')
    }
  }

  const handleTaskChange = useCallback(
    (taskId) => {
      const selectedTask = TASK_OPTIONS.find((task) => task.id === taskId)
      if (!selectedTask) return
      setSelectedTaskId(taskId)
      setStatusMessage('Updating task...')
      setModelConfig((prev) => {
        const nextConfig = { ...prev, ...taskConfigFromSelection(selectedTask) }
        persistModelConfig(nextConfig)
          .then((savedConfig) => {
            setModelConfig((current) => ({
              ...current,
              ...savedConfig,
              taskDescription:
                savedConfig.task_description ??
                savedConfig.taskDescription ??
                selectedTask.description,
              task_id: selectedTask.id,
            }))
            setStatusMessage('Task updated')
          })
          .catch((err) => {
            setStatusMessage(`Failed to update task: ${err.message}`)
          })
        return nextConfig
      })
    },
    [persistModelConfig],
  )

  const saveModelConfig = async () => {
    setStatusMessage('Updating model config...')
    try {
      const savedConfig = await persistModelConfig(modelConfig)
      setModelConfig((prev) => ({
        ...prev,
        ...savedConfig,
        taskDescription:
          savedConfig.task_description ?? savedConfig.taskDescription ?? prev.taskDescription,
      }))
      setStatusMessage('Model config saved')
    } catch (err) {
      setStatusMessage(`Failed to update config: ${err.message}`)
    }
  }

  useEffect(
    () => () => {
      if (call) call.leave().catch(() => {})
      if (client) client.disconnectUser().catch(() => {})
    },
    [call, client],
  )

  const appendTranscriptEntry = useCallback(
    (entry) => {
      setTranscriptEntries((prev) => [...prev, createTranscriptEntry(entry)].slice(-80))
    },
    [createTranscriptEntry],
  )

  // Poll the backend session-chat store every 3 s while a session is active and append new entries.
  useEffect(() => {
    if (!callSession) return undefined
    const callId = callSession.callId
    let seenCount = 0
    const interval = setInterval(async () => {
      try {
        const r = await fetch(`${API_BASE_URL}/api/session/${callId}/session-chat`)
        if (!r.ok) return
        const data = await r.json()
        const entries = data.transcript ?? []
        if (entries.length <= seenCount) return
        const newEntries = entries.slice(seenCount)
        seenCount = entries.length
        newEntries.forEach((t) =>
          appendTranscriptEntry({
            kind: 'message',
            speaker: t.speaker,
            role: normalizeSpeakerRole(t.speaker),
            text: t.text,
            // Backend timestamp is a string label; stamp with current unix ms on receipt.
            timestamp: Date.now(),
          }),
        )
      } catch {
        // ignore poll errors
      }
    }, 3000)
    return () => clearInterval(interval)
  }, [callSession, appendTranscriptEntry])

  const isConnected = Boolean(client && call && callSession)

  if (!currentUser) {
    return <LoginPage onLogin={handleLogin} />
  }

  if (!isConnected) {
    return (
      <PreSessionContent
        onStart={startCall}
        isStarting={isStarting}
        statusMessage={statusMessage}
        modelConfig={modelConfig}
        setModelConfig={setModelConfig}
        saveModelConfig={saveModelConfig}
        transcriptEntries={transcriptEntries}
        isTranscriptVisible={isTranscriptVisible}
        onToggleTranscript={() => setIsTranscriptVisible((prev) => !prev)}
        selectedTaskId={selectedTaskId}
        onTaskChange={handleTaskChange}
        tasksCatalog={tasksCatalog}
        isStatsVisible={isStatsVisible}
        onToggleStats={() => setIsStatsVisible((prev) => !prev)}
      />
    )
  }

  return (
    <StreamVideo client={client}>
      <StreamCall call={call}>
        <CallContent
          callSession={callSession}
          onLeave={leaveCall}
          onStart={startCall}
          isStarting={isStarting}
          isConnected={isConnected}
          statusMessage={statusMessage}
          setStatusMessage={setStatusMessage}
          modelConfig={modelConfig}
          setModelConfig={setModelConfig}
          saveModelConfig={saveModelConfig}
          transcriptEntries={transcriptEntries}
          appendTranscriptEntry={appendTranscriptEntry}
          isTranscriptVisible={isTranscriptVisible}
          onToggleTranscript={() => setIsTranscriptVisible((prev) => !prev)}
          selectedTaskId={selectedTaskId}
          onTaskChange={handleTaskChange}
          tasksCatalog={tasksCatalog}
          isStatsVisible={isStatsVisible}
          onToggleStats={() => setIsStatsVisible((prev) => !prev)}
        />
      </StreamCall>
    </StreamVideo>
  )
}

export default App
