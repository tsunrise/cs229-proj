use pulldown_cmark::Parser;
use pyo3::prelude::*;
use rayon::prelude::*;

fn postprocess_text(s: &str) -> String {
    // only allow alphanumeric characters, spaces, newlines, and some punctuation
    const SPECIAL_CHARS: &str = ".,;:!?()[]{}<>\"'+-*/";
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric() || c.is_whitespace() || SPECIAL_CHARS.contains(*c))
        .collect()
}

fn markdown_2_text_inner(s: &str) -> String {
    let parser = Parser::new(s);
    let mut text = String::new();
    for event in parser {
        match event {
            pulldown_cmark::Event::Text(s) => {
                text.push_str(&s);
                text.push(' ')
            }
            pulldown_cmark::Event::SoftBreak => text.push(' '),
            pulldown_cmark::Event::HardBreak => text.push(' '),
            _ => (),
        }
    }
    postprocess_text(&text)
}

#[pyfunction]
#[pyo3(text_signature = "(text)")]
fn markdown_to_text(s: &str) -> PyResult<String> {
    Ok(markdown_2_text_inner(s))
}

#[pyfunction]
fn batch_markdown_to_text(s: Vec<&str>) -> PyResult<Vec<String>> {
    Ok(s.par_iter().map(|s| markdown_2_text_inner(s)).collect())
}

#[pyfunction]
fn normalize_text_simple(s: &str) -> PyResult<String> {
    Ok(postprocess_text(s))
}

#[pyfunction]
fn batch_normalize_text_simple(s: Vec<&str>) -> PyResult<Vec<String>> {
    Ok(s.par_iter().map(|s| postprocess_text(s)).collect())
}

#[pymodule]
fn md2txt(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(markdown_to_text))?;
    m.add_wrapped(wrap_pyfunction!(batch_markdown_to_text))?;
    m.add_wrapped(wrap_pyfunction!(normalize_text_simple))?;
    m.add_wrapped(wrap_pyfunction!(batch_normalize_text_simple))?;
    Ok(())
}
