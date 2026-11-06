//! Chat template rendering for the template families this build recognises.
//!
//! A chat template is a Jinja program, and a model was trained on the exact text it produces. Running
//! Jinja here is out of scope, and approximating an arbitrary template would silently feed the model
//! text it has never seen. So this module does the narrow, checkable thing: it recognises a template
//! by its distinctive markers, renders the family's canonical format, and refuses anything it does
//! not recognise. Refusing is a supported outcome, not a failure mode.
//!
//! Recognised families:
//!
//! - **ChatML** (`<|im_start|>`): Qwen, Hermes, and most instruction-tuned derivatives.
//! - **Llama 3** (`<|start_header_id|>`).
//! - **Llama 2** (`[INST]` with `<<SYS>>`).
//! - **Mistral** (`[INST]` without `<<SYS>>`).
//!
//! ChatML output is checked against a real model on this machine (see the crate tests); the others
//! follow the canonical format published with those model families.

use vipera_errors::ViperaError;

/// Who produced a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Instructions for the whole conversation.
    System,
    /// Input from the person using the model.
    User,
    /// A previous model reply, kept for context.
    Assistant,
}

impl Role {
    /// Name used in the rendered text.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }

    /// Parse the role name accepted on the command line and in request payloads.
    pub fn parse(name: &str) -> Result<Self, ViperaError> {
        match name {
            "system" => Ok(Self::System),
            "user" => Ok(Self::User),
            "assistant" => Ok(Self::Assistant),
            other => Err(ViperaError::invalid_argument(format!(
                "unknown role \"{other}\"; use system, user, or assistant"
            ))),
        }
    }
}

/// One turn of a conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    /// Who said it.
    pub role: Role,
    /// Text of the message, verbatim.
    pub content: String,
}

impl ChatMessage {
    /// Build a message.
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }

    /// Build a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }

    /// Build a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }

    /// Build an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content)
    }
}

/// A template family whose rendering this build knows exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateFamily {
    /// `<|im_start|>role\ncontent<|im_end|>\n`
    ChatMl,
    /// `<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>`
    Llama3,
    /// `<s>[INST] ... [/INST] ... </s>` with the system prompt inside `<<SYS>>`.
    Llama2,
    /// `<s>[INST] ... [/INST]` with no system slot.
    Mistral,
}

impl TemplateFamily {
    /// Name used in diagnostics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ChatMl => "chatml",
            Self::Llama3 => "llama3",
            Self::Llama2 => "llama2",
            Self::Mistral => "mistral",
        }
    }
}

/// Identify a template family from the markers its text contains.
///
/// Returns `None` for templates this build cannot render exactly, including Gemma, whose templates
/// differ enough between model versions that one canonical rendering would be wrong for some of them.
pub fn detect_family(template: &str) -> Option<TemplateFamily> {
    if template.contains("<|im_start|>") {
        return Some(TemplateFamily::ChatMl);
    }
    if template.contains("<|start_header_id|>") {
        return Some(TemplateFamily::Llama3);
    }
    if template.contains("[INST]") {
        return if template.contains("<<SYS>>") {
            Some(TemplateFamily::Llama2)
        } else {
            Some(TemplateFamily::Mistral)
        };
    }
    None
}

/// Render `messages` with the model's own template.
///
/// `add_generation_prompt` appends the assistant header that tells the model to start replying, which
/// is what a generation request needs and what a token-count comparison does not.
pub fn render(
    template: &str,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
) -> Result<String, ViperaError> {
    let family = detect_family(template).ok_or_else(|| {
        let first_line = template.lines().next().unwrap_or("").trim();
        let hint = if template.contains("<start_of_turn>") {
            "; Gemma templates vary by model version and are not rendered by this build"
        } else {
            ""
        };
        ViperaError::unsupported(format!(
            "chat template is not one this build renders (recognised: chatml, llama3, llama2, mistral); \
             template starts with {first_line:?}{hint}"
        ))
    })?;
    render_family(family, messages, add_generation_prompt)
}

/// Render `messages` in a known family's canonical format.
pub fn render_family(
    family: TemplateFamily,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
) -> Result<String, ViperaError> {
    validate(family, messages)?;
    Ok(match family {
        TemplateFamily::ChatMl => render_chatml(messages, add_generation_prompt),
        TemplateFamily::Llama3 => render_llama3(messages, add_generation_prompt),
        TemplateFamily::Llama2 => render_llama2(messages, add_generation_prompt),
        TemplateFamily::Mistral => render_mistral(messages, add_generation_prompt),
    })
}

fn validate(family: TemplateFamily, messages: &[ChatMessage]) -> Result<(), ViperaError> {
    if messages.is_empty() {
        return Err(ViperaError::invalid_argument(
            "a chat request needs at least one message",
        ));
    }
    let first_non_system = messages.iter().find(|m| m.role != Role::System);
    match first_non_system {
        None => {
            return Err(ViperaError::invalid_argument(
                "a chat request needs a user message, not only a system prompt",
            ));
        }
        Some(first) if first.role != Role::User => {
            return Err(ViperaError::invalid_argument(
                "a chat request must start with a user message, not with a model reply",
            ));
        }
        Some(_) => {}
    }
    if messages
        .iter()
        .enumerate()
        .any(|(index, message)| message.role == Role::System && index != 0)
    {
        return Err(ViperaError::invalid_argument(
            "a system message must come first; later system messages cannot be rendered",
        ));
    }
    if family == TemplateFamily::Mistral && messages.iter().any(|m| m.role == Role::System) {
        return Err(ViperaError::invalid_argument(
            "the mistral template has no system slot; merge the instructions into the user message",
        ));
    }
    Ok(())
}

fn render_chatml(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
    let mut out = String::new();
    for message in messages {
        out.push_str("<|im_start|>");
        out.push_str(message.role.as_str());
        out.push('\n');
        out.push_str(&message.content);
        out.push_str("<|im_end|>\n");
    }
    if add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
    }
    out
}

fn render_llama3(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
    let mut out = String::new();
    for message in messages {
        out.push_str("<|start_header_id|>");
        out.push_str(message.role.as_str());
        out.push_str("<|end_header_id|>\n\n");
        // The published Llama 3 template trims message content before writing it.
        out.push_str(message.content.trim());
        out.push_str("<|eot_id|>");
    }
    if add_generation_prompt {
        out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    }
    out
}

fn render_llama2(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
    // The assistant header is the `[/INST]` marker itself, so there is nothing to append.
    let _ = add_generation_prompt;
    let system = messages
        .first()
        .filter(|m| m.role == Role::System)
        .map(|m| m.content.as_str());
    let turns: Vec<&ChatMessage> = messages.iter().filter(|m| m.role != Role::System).collect();

    let mut out = String::from("<s>");
    let mut first_instruction = true;
    for message in turns {
        match message.role {
            Role::User => {
                if !first_instruction {
                    out.push_str("<s>");
                }
                out.push_str("[INST] ");
                if let (true, Some(system)) = (first_instruction, system) {
                    out.push_str("<<SYS>>\n");
                    out.push_str(system);
                    out.push_str("\n<</SYS>>\n\n");
                }
                out.push_str(&message.content);
                out.push_str(" [/INST]");
                first_instruction = false;
            }
            Role::Assistant => {
                out.push(' ');
                out.push_str(&message.content);
                out.push_str(" </s>");
            }
            Role::System => {}
        }
    }
    out
}

fn render_mistral(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
    // As with Llama 2, `[/INST]` already opens the assistant turn.
    let _ = add_generation_prompt;
    let mut out = String::from("<s>");
    for message in messages {
        match message.role {
            Role::User => {
                out.push_str("[INST] ");
                out.push_str(&message.content);
                out.push_str(" [/INST]");
            }
            Role::Assistant => {
                out.push(' ');
                out.push_str(&message.content);
                out.push_str("</s>");
            }
            Role::System => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHATML: &str = "{%- for message in messages %}{{- '<|im_start|>' + message['role'] }}\
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}";
    const LLAMA3: &str = "<|start_header_id|>{{ message['role'] }}<|end_header_id|>";
    const LLAMA2: &str = "{% if message['role'] == 'user' %}[INST] <<SYS>>";
    const MISTRAL: &str = "{% for message in messages %}[INST] {{ message['content'] }} [/INST]";

    #[test]
    fn detects_each_family_by_its_marker() {
        assert_eq!(detect_family(CHATML), Some(TemplateFamily::ChatMl));
        assert_eq!(detect_family(LLAMA3), Some(TemplateFamily::Llama3));
        assert_eq!(detect_family(LLAMA2), Some(TemplateFamily::Llama2));
        assert_eq!(detect_family(MISTRAL), Some(TemplateFamily::Mistral));
        assert_eq!(detect_family("{% for m in messages %}Hi{% endfor %}"), None);
    }

    #[test]
    fn chatml_matches_the_qwen_shape() {
        let messages = [ChatMessage::system("be brief"), ChatMessage::user("hi")];
        assert_eq!(
            render(CHATML, &messages, true).unwrap(),
            "<|im_start|>system\nbe brief<|im_end|>\n\
             <|im_start|>user\nhi<|im_end|>\n\
             <|im_start|>assistant\n"
        );
        assert_eq!(
            render(CHATML, &messages, false).unwrap(),
            "<|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n"
        );
    }

    #[test]
    fn llama3_uses_header_blocks_and_trims_content() {
        let messages = [
            ChatMessage::system(" be brief\n"),
            ChatMessage::user("hi"),
            ChatMessage::assistant("hello"),
        ];
        assert_eq!(
            render(LLAMA3, &messages, true).unwrap(),
            "<|start_header_id|>system<|end_header_id|>\n\nbe brief<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\nhello<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn llama2_puts_the_system_prompt_inside_the_first_instruction() {
        let messages = [
            ChatMessage::system("be brief"),
            ChatMessage::user("hi"),
            ChatMessage::assistant("hello"),
            ChatMessage::user("again"),
        ];
        assert_eq!(
            render(LLAMA2, &messages, true).unwrap(),
            "<s>[INST] <<SYS>>\nbe brief\n<</SYS>>\n\nhi [/INST] hello </s><s>[INST] again [/INST]"
        );
    }

    #[test]
    fn mistral_has_no_system_slot_and_ends_turns_with_eos() {
        let messages = [
            ChatMessage::user("hi"),
            ChatMessage::assistant("hello"),
            ChatMessage::user("again"),
        ];
        assert_eq!(
            render(MISTRAL, &messages, true).unwrap(),
            "<s>[INST] hi [/INST] hello</s>[INST] again [/INST]"
        );
        let with_system = [ChatMessage::system("x"), ChatMessage::user("hi")];
        assert!(render(MISTRAL, &with_system, true).is_err());
    }

    #[test]
    fn an_unknown_template_is_refused_with_a_reason() {
        let err = render(
            "{% for m in messages %}{{ m.content }}{% endfor %}",
            &[ChatMessage::user("hi")],
            true,
        )
        .unwrap_err();
        assert!(err.to_string().contains("not one this build renders"));
        assert_eq!(err.exit_code(), 2);
    }

    #[test]
    fn gemma_is_named_as_unsupported_rather_than_approximated() {
        let err = render("<start_of_turn>user", &[ChatMessage::user("hi")], true).unwrap_err();
        assert!(err.to_string().contains("Gemma"));
    }

    #[test]
    fn invalid_conversations_are_rejected() {
        assert!(render(CHATML, &[], true).is_err());
        assert!(render(CHATML, &[ChatMessage::system("only")], true).is_err());
        assert!(render(CHATML, &[ChatMessage::assistant("first")], true).is_err());
        assert!(
            render(
                CHATML,
                &[ChatMessage::user("hi"), ChatMessage::system("too late"),],
                true
            )
            .is_err()
        );
    }

    #[test]
    fn roles_parse_and_round_trip() {
        for role in [Role::System, Role::User, Role::Assistant] {
            assert_eq!(Role::parse(role.as_str()).unwrap(), role);
        }
        assert!(Role::parse("tool").is_err());
    }
}
