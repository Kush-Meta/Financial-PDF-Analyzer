# FinRead design memo

Make financial research intuitive to talk to and enjoyable to use. The interface
should invite curiosity while making the source of every answer easy to inspect.

## Reference projects

- [NotebookLM design case study by Jason Spielman](https://jasonspielman.com/notebooklm):
  adjacent source, conversation, and output panels reduce context switching.
  FinRead applies that idea with a source sidebar, central conversation, and evidence desk.
- [Open WebUI follow-up prompts](https://docs.openwebui.com/features/chat-conversations/chat-features/follow-up-prompts/):
  suggestions provide a next step after an answer. FinRead offered explicit, fixed
  follow-up actions in an earlier iteration. The current design intentionally removes
  suggestions and prioritizes free-form conversation per user feedback.
- [LM Studio chat interface](https://lmstudio.ai/docs/app/basics/chat):
  a focused conversation surface with supporting controls. FinRead moves model
  selection and retrieval configuration into a secondary sidebar expander.

These are interaction references, not copied assets or product integrations.

## Experience

- Start with a source. Apple's complete FY2025 SEC filing makes the experience usable
  before the user has a PDF ready. Use real public filings for product examples;
  reserve synthetic documents for automated tests. Model calls start only after a question.
- Make the question box the focal point. Do not add generic starter cards or
  automatic follow-up buttons; let users ask in their own words.
- Keep the conversation central. Source chips select evidence alongside the answer;
  opening the full page does not trigger another model call.
- Keep document tools collapsed before the first answer and group exports and
  conversation management under More. Show source controls when evidence is available.
- Recover gracefully. A failed question remains available for one-click retry, and
  a new conversation preserves the existing document index.
- Let research leave the app as JSON or readable notes with source excerpts.

## Visual language

Use a plain finread wordmark with no letter-in-a-blue-square symbol. Reserve
cobalt for the primary action and composer focus; use white working surfaces and
navy text. Omit decorative badges, slogans, and feature grids. Native Streamlit controls retain their
keyboard and accessibility semantics. Styling includes visible focus treatment,
small-screen rules, and reduced-motion support.

Keep the distinction between reported evidence and model interpretation visible.
No fabricated portfolios, fake recent documents, placeholder answers, or decorative
financial metrics. Do not replace the working Python/model pipeline merely to change the UI.

## Psychology and usability principles

Research reviewed September 13, 2026. These principles guide design decisions;
they do not establish that this particular interface has passed a usability study.

- **Attention and visual hierarchy:** size, contrast, and placement communicate
  importance. Reserve the strongest emphasis for the current task: opening a
  filing first, then asking a question. Avoid making every control compete with
  the composer. [NN/G: visual hierarchy](https://www.nngroup.com/articles/visual-hierarchy-ux-definition/)
- **Progressive disclosure:** keep frequent actions visible and reveal less-used
  tools when requested. Hiding controls is only helpful when people can still
  predict where to find them. Settings and document tools can be secondary;
  source references should remain obvious next to answers.
  [NN/G: progressive disclosure](https://www.nngroup.com/articles/progressive-disclosure/)
- **Recognition over recall:** retain the current filing name, readable labels,
  and page references so users need not remember the document or provenance.
  Simplicity should reduce memory demands, not replace useful labels with
  unexplained icons. [NN/G: recognition and recall](https://www.nngroup.com/articles/recognition-and-recall/)
- **Feedback and recovery:** visible progress, preserved questions, and retry
  actions help people understand what happened and continue after a failure.
  [NN/G: usability heuristics](https://www.nngroup.com/articles/ten-usability-heuristics/)

Next refinements should be driven by observed tasks: can a first-time user open
a filing, ask a question, inspect its source, and save the result without guidance?
In particular, test whether the label More makes exports sufficiently discoverable
and whether users notice the source chips. Prefer a clearer label or placement
before adding another button. Keep this as a research agenda, not a claim of measured improvement.
