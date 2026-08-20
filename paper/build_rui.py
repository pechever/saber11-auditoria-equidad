# Construye la version RUI del articulo EDITANDO la plantilla oficial
# V4 Plantilla RUI Espanol 2026.docx en su sitio: conserva la primera pagina
# (cuadro lateral con tipologia, ISSN y licencia; titulos; bloque de autores;
# fechas; Resumen / Palabras clave / Abstract / Keywords) y sustituye el
# cuerpo de ejemplo por el contenido del articulo.
import re, zipfile, shutil
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

SRC = 'rui_contenido.md'
OUT = 'Auditoria_equidad_Saber11_RUI.docx'
IMGMAP = {'f1': 'g_rui/f1.png', 'f2': 'g_rui/f2.png', 'f4': 'g_rui/f4.png',
          'f5': 'g_rui/f5.png', 'f5b': 'g_rui/f6.png'}
TXBX = qn('w:txbxContent')

# ------------------------------------------------------------------ contenido
front, body = {}, []
en_cuerpo = False
for ln in open(SRC, encoding='utf-8').read().split('\n'):
    ln = ln.rstrip()
    if not ln:
        continue
    tag, _, txt = ln.partition(' ')
    if tag == '#H1':
        en_cuerpo = True
    if en_cuerpo:
        body.append((tag, txt))
    else:
        front[tag] = txt

doc = Document('plantilla_rui.docx')
styles = {s.name for s in doc.styles}
def S(*n):
    for x in n:
        if x in styles:
            return x
    return 'Normal'
ST_FIG, ST_TAB = S('Figura Título'), S('Tabla Título')
ST_SUB = S('Subtitulos')

# --------------------------------------------------- utilidades de escritura
def fijar(run, texto):
    run.text = texto
    if run.font.name is None:
        run.font.name = 'Verdana'

def reemplazar_propio(p, texto):
    """Sustituye el texto de un parrafo conservando el formato del primer run
    y sin tocar el contenido de cuadros de texto anclados en el."""
    propios = [r for r in p.runs
               if not any(a.tag == TXBX for a in _ancestros(r._element))]
    if not propios:
        return
    fijar(propios[0], texto)
    for r in propios[1:]:
        r.text = ''

def _ancestros(el):
    a = el.getparent()
    while a is not None:
        yield a
        a = a.getparent()

def texto_en_cuadro(p, viejo, nuevo):
    for t in p._p.iter(qn('w:t')):
        if t.text and viejo in t.text:
            t.text = t.text.replace(viejo, nuevo)
            return True
    return False

# ------------------------------------------------------ 1. primera pagina
P = doc.paragraphs
# altura estimada del bloque de titulos (20 pt, ~48 caracteres por linea)
def _lineas(t, por_linea=42):
    return max(1, -(-len(t) // por_linea))
TITULO_ALTO_CM = 0.9 + 0.82 * (_lineas(front['#T1']) + _lineas(front['#T2']))
reemplazar_propio(P[0], front['#T1'])
reemplazar_propio(P[1], front['#T2'])

# autores: nombre + numero de filiacion en superindice + asterisco
runs4 = [r for r in P[4].runs
         if not any(a.tag == TXBX for a in _ancestros(r._element))]
plan = [front['#AU'], '1', '', '', '', '*', '', '']
for r, val in zip(runs4[1:], plan):
    r.text = val
for r in runs4[1 + len(plan):]:
    r.text = ''

# tipologia dentro del cuadro lateral
texto_en_cuadro(P[4], 'Artículo de Investigación, reporte de caso, estado del arte',
                front['#TIPO'])

# filiacion: se conserva el superindice inicial y se reescribe el resto
runs7 = P[7].runs
fijar(runs7[2], front['#AF'])
for r in runs7[3:]:
    r.text = ''

reemplazar_propio(P[15], front['#RES'])
reemplazar_propio(P[19], front['#KWES'])
reemplazar_propio(P[23], front['#ABS'])
reemplazar_propio(P[28], front['#KWEN'])

# El pie "Como citar" de la portada es alto e invade el texto que llegue al
# fondo de la pagina. Se lleva el bloque en ingles a la pagina siguiente.
P[21].paragraph_format.page_break_before = True

# sobran dos lineas de filiacion (2 y 3 autores) y una linea de ejemplo del
# abstract; se eliminan de atras hacia adelante para no desplazar indices
for idx in (24, 9, 8):
    el = P[idx]._element
    el.getparent().remove(el)

# ------------------------------------------------- 2. borrar cuerpo de ejemplo
body_el = doc.element.body
sectPr = body_el.find(qn('w:sectPr'))
inicio = None
for el in list(body_el):
    if el.tag == qn('w:p'):
        ps = el.find(qn('w:pPr'))
        st = ps.find(qn('w:pStyle')) if ps is not None else None
        txt = ''.join(t.text or '' for t in el.iter(qn('w:t')))
        if st is not None and st.get(qn('w:val')) == 'Ttulo1' and 'Introducción' in txt:
            inicio = el
            break
borrar, visto = [], False
for el in list(body_el):
    if el is inicio:
        visto = True
    if visto and el is not sectPr:
        borrar.append(el)
for el in borrar:
    body_el.remove(el)

# ------------------------------------------------------------- 3. cuerpo nuevo
def _runs(p, text, size=None, bold=None, italic=None, markup=True):
    # _cursiva_ para extranjerismos y terminos clave (unico uso permitido).
    # Se desactiva en referencias: las URL contienen guiones bajos.
    partes = re.split(r'(_[^_]+_)', text) if markup else [text]
    for part in partes:
        if not part:
            continue
        it = markup and part.startswith('_') and part.endswith('_') and len(part) > 2
        r = p.add_run(part[1:-1] if it else part)
        r.font.name = 'Verdana'
        r._element.rPr.rFonts.set(qn('w:eastAsia'), 'Verdana')
        if size is not None: r.font.size = Pt(size)
        if bold is not None: r.font.bold = bold
        if it: r.font.italic = True
        elif italic is not None: r.font.italic = italic

def add(text='', style='Normal', align=None, size=None, bold=None, italic=None,
        before=None, after=None, keep=False, markup=True):
    p = doc.add_paragraph(style=style)
    _runs(p, text, size=size, bold=bold, italic=italic, markup=markup)
    if align is not None:
        p.alignment = align
    pf = p.paragraph_format
    pf.line_spacing = 1.0
    if before is not None: pf.space_before = Pt(before)
    if after is not None:  pf.space_after = Pt(after)
    if keep: pf.keep_with_next = True
    return p

def titulo(text, level):
    # numeracion manual; se anula la automatica del estilo para no duplicarla
    p = doc.add_paragraph(style=f'Heading {level}')
    pPr = p._p.get_or_add_pPr()
    numPr = OxmlElement('w:numPr')
    for nom, val in (('w:ilvl', '0'), ('w:numId', '0')):
        e = OxmlElement(nom); e.set(qn('w:val'), val); numPr.append(e)
    pPr.append(numPr)
    _runs(p, text, bold=True)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format
    pf.line_spacing = 1.0; pf.space_before = Pt(10); pf.space_after = Pt(6)
    pf.keep_with_next = True

def tabla(rows):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j); c.text = ''
            p = c.paragraphs[0]
            _runs(p, val, size=8, bold=(i == 0))
            p.alignment = (WD_ALIGN_PARAGRAPH.CENTER if (i == 0 or j > 0)
                           else WD_ALIGN_PARAGRAPH.LEFT)
            pf = p.paragraph_format
            pf.line_spacing = 1.0; pf.space_after = Pt(2); pf.space_before = Pt(2)
            # la tabla no se parte entre paginas y arrastra consigo su fuente
            pf.keep_with_next = True
        t.rows[i]._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
    b = OxmlElement('w:tblBorders')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        e = OxmlElement(f'w:{edge}')
        if edge in ('top', 'bottom'):
            e.set(qn('w:val'), 'single'); e.set(qn('w:sz'), '8')
        else:
            e.set(qn('w:val'), 'none'); e.set(qn('w:sz'), '0')
        e.set(qn('w:color'), '000000'); b.append(e)
    t._tbl.tblPr.append(b)
    for cell in t.rows[0].cells:
        tb = OxmlElement('w:tcBorders'); bo = OxmlElement('w:bottom')
        bo.set(qn('w:val'), 'single'); bo.set(qn('w:sz'), '8'); bo.set(qn('w:color'), '000000')
        tb.append(bo); cell._tc.get_or_add_tcPr().append(tb)
    t.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))

n_pal = 0
i = 0
while i < len(body):
    tag, txt = body[i]
    if tag == '#H1':
        titulo(txt, 1)
    elif tag == '#H2':
        titulo(txt, 2)
    elif tag == '#SUB':
        add(txt, ST_SUB, WD_ALIGN_PARAGRAPH.LEFT, bold=True, before=10, after=6, keep=True)
    elif tag == '#P':
        add(txt, 'Normal', WD_ALIGN_PARAGRAPH.JUSTIFY, after=6); n_pal += len(txt.split())
    elif tag == '#B':
        p = add('•  ' + txt, 'Normal', WD_ALIGN_PARAGRAPH.JUSTIFY, after=4)
        p.paragraph_format.left_indent = Cm(0.8)
        n_pal += len(txt.split())
    elif tag == '#IMG':
        key, w, _h = txt.split('|')
        p = doc.add_paragraph(style='Normal')
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf = p.paragraph_format
        pf.space_before = Pt(8); pf.space_after = Pt(4); pf.keep_with_next = True
        p.add_run().add_picture(IMGMAP[key], width=Pt(int(w)))
    elif tag == '#FIGC':
        add(txt, ST_FIG, WD_ALIGN_PARAGRAPH.CENTER, size=10, after=2, keep=True)
    elif tag == '#TABC':
        add(txt, ST_TAB, WD_ALIGN_PARAGRAPH.CENTER, size=10, before=8, after=4, keep=True)
    elif tag == '#SRC':
        add(txt, 'Normal', WD_ALIGN_PARAGRAPH.CENTER, size=8, italic=True, after=10)
    elif tag == '#TAB':
        bloque = []
        while i < len(body) and body[i][0] == '#TAB':
            bloque.append([c.strip() for c in body[i][1].split('|')]); i += 1
        tabla(bloque); continue
    elif tag == '#R':
        p = add(txt, 'Normal', WD_ALIGN_PARAGRAPH.JUSTIFY, size=9, after=4, markup=False)
        p.paragraph_format.left_indent = Cm(0.9)
        p.paragraph_format.first_line_indent = Cm(-0.9)
    i += 1

doc.save(OUT)

# ------------------------------------- 4. marcadores de encabezado y pie
TIT_CORTO = 'Auditoría de equidad de modelos de aprendizaje automático'
SUST = [('Título del artículo', TIT_CORTO),
        ('Apellido, A. Apellido', 'F. Echeverría'),
        ('Autores (A. Apellido).', 'F. Echeverría'),
        ('Autores (A. Apellido)', 'F. Echeverría'),
        ('Título en el idioma en que se presenta el artículo', front['#T1'])]
# El cuadro lateral (tipologia, ISSN, licencia) esta anclado al parrafo de
# autores. Con un titulo de cuatro lineas invade la zona del titulo, asi que
# se ancla al margen superior, por debajo del bloque de titulos, y se recorta
# su altura para que no alcance el pie de pagina.
EMU = 360000                      # 1 cm
TOP_CAJA = round(TITULO_ALTO_CM * EMU)
ALTO_UTIL_CM = 27.94 - 2.0 - 2.0        # alto de la caja de texto de la pagina
# se dejan 3,5 cm libres al pie para no tapar el bloque "Como citar"
ALTO_CAJA = round((ALTO_UTIL_CM - TITULO_ALTO_CM - 3.5) * EMU)
ANCLA_VIEJA = ('<wp:positionV relativeFrom="paragraph">'
               '<wp:posOffset>97790</wp:posOffset></wp:positionV>')
ANCLA_NUEVA = ('<wp:positionV relativeFrom="margin">'
               f'<wp:posOffset>{TOP_CAJA}</wp:posOffset></wp:positionV>')

n_h = 0
tmp = OUT + '.tmp'
with zipfile.ZipFile(OUT) as zin, zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename == 'word/document.xml':
            s = data.decode('utf-8')
            s = s.replace(ANCLA_VIEJA, ANCLA_NUEVA)
            s = s.replace('cx="1504950" cy="6943725"', f'cx="1504950" cy="{ALTO_CAJA}"')
            data = s.encode('utf-8')
        elif item.filename.startswith(('word/header', 'word/footer')) and item.filename.endswith('.xml'):
            s = antes = data.decode('utf-8')
            for viejo, nuevo in SUST:
                s = s.replace(viejo, nuevo)
            # elimina la numeracion automatica que anteponia "A." al nombre
            s = re.sub(r'<w:numPr>.*?</w:numPr>', '', s)
            if s != antes:
                n_h += 1
            data = s.encode('utf-8')
        zout.writestr(item, data)
shutil.move(tmp, OUT)

# ------------------------------------------------------------ verificacion
d2 = Document(OUT)
res = [p.text for p in d2.paragraphs]
print('archivo:', OUT)
print('encabezados/pies corregidos:', n_h)
print('titulo ES palabras:', len(front['#T1'].split()), ' EN:', len(front['#T2'].split()))
print('resumen ES palabras:', len(front['#RES'].split()), '(limite 150)')
print('abstract EN palabras:', len(front['#ABS'].split()), '(limite 150)')
print('palabras clave ES:', len(front['#KWES'].split(';')), ' EN:', len(front['#KWEN'].split(';')))
print('palabras cuerpo aprox:', n_pal, '(limite 12.000)')
print('tablas:', len(d2.tables), ' figuras:',
      sum(1 for p in d2.paragraphs if p.style.name == ST_FIG))
sec = d2.sections[0]
print('margenes cm: %.1f %.1f %.1f %.1f' % (sec.top_margin.cm, sec.bottom_margin.cm,
                                            sec.left_margin.cm, sec.right_margin.cm))
for marcador in ('Apellido', 'Grupo de investigación', 'dd mes aaaa', 'CCTO'):
    print(f'  marcador "{marcador}" en el cuerpo:', any(marcador in t for t in res))
