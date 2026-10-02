const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, AlignmentType, BorderStyle, ImageRun, PageBreak,
  LevelFormat, convertInchesToTwip, VerticalAlign, TableOfContents
} = require("docx");
const fs = require("fs");

// ---------- helpers ----------
function P(text, opts = {}) {
  const { bold, italic, size, spacingAfter = 160, alignment, spacingBefore } = opts;
  return new Paragraph({
    alignment,
    spacing: { after: spacingAfter, before: spacingBefore },
    children: [new TextRun({ text, bold, italics: italic, size })],
  });
}

// Rich paragraph: array of {text, bold, italic, sup} runs
function RP(parts, opts = {}) {
  const { spacingAfter = 160, alignment, spacingBefore } = opts;
  return new Paragraph({
    alignment,
    spacing: { after: spacingAfter, before: spacingBefore },
    children: parts.map(p => new TextRun({
      text: p.text, bold: p.bold, italics: p.italic, superScript: p.sup, size: p.size
    })),
  });
}

function H1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 160 }, children: [new TextRun({ text })] });
}
function H2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 260, after: 140 }, children: [new TextRun({ text })] });
}
function H3(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_3, spacing: { before: 200, after: 100 }, children: [new TextRun({ text, italics: true })] });
}

function caption(text) {
  return new Paragraph({
    spacing: { before: 80, after: 240 },
    children: [new TextRun({ text, italics: true, size: 20 })],
  });
}

function cell(text, opts = {}) {
  const { bold, width, shade, align = AlignmentType.LEFT, size = 20 } = opts;
  return new TableCell({
    width: width ? { size: width, type: WidthType.DXA } : undefined,
    shading: shade ? { type: ShadingType.CLEAR, fill: shade } : undefined,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ alignment: align, children: [new TextRun({ text: String(text), bold, size })] })],
  });
}

function makeTable(headers, rows, widths) {
  const totalWidth = widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({
        tableHeader: true,
        children: headers.map((h, i) => cell(h, { bold: true, width: widths[i], shade: "D9E2F3", align: AlignmentType.CENTER })),
      }),
      ...rows.map(r => new TableRow({
        children: r.map((c, i) => cell(c, { width: widths[i], align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })),
      })),
    ],
  });
}

function img(path, w, h) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 160, after: 40 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path), transformation: { width: w, height: h } })],
  });
}

const UP = "/mnt/user-data/uploads/TinyML/";

module.exports = { Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, AlignmentType, BorderStyle, ImageRun, PageBreak, VerticalAlign,
  P, RP, H1, H2, H3, caption, cell, makeTable, img, UP, convertInchesToTwip };
