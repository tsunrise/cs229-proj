use pulldown_cmark::Parser;
use pyo3::prelude::*;
use rayon::prelude::*;

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
    text
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

#[pymodule]
fn md2txt(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(markdown_to_text))?;
    m.add_wrapped(wrap_pyfunction!(batch_markdown_to_text))?;
    Ok(())
}
