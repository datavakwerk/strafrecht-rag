from datetime import date
from pathlib import Path

from strafrecht_rag.ingest.model import Uitspraak
from strafrecht_rag.ingest.parser import ParseError, parse_dir, parse_file, parse_xml

FIXTURES = Path(__file__).parent / "fixtures"


def test_rechtbank_metadata():
    u = parse_xml((FIXTURES / "ecli_rbnne_2024_674.xml").read_bytes())
    assert u.ecli == "ECLI:NL:RBNNE:2024:674"
    assert u.instantie == "Rechtbank Noord-Nederland"
    assert u.instantie_type == "Rechtbank"
    assert u.datum == date(2024, 3, 1)
    assert u.rechtsgebieden == ["Strafrecht"]
    assert "Eerste aanleg - meervoudig" in u.procedures
    assert u.zaaknummer == "18.283374.23"
    assert u.wetsverwijzingen == ["Wetboek van Strafrecht 312"]
    assert u.samenvatting.startswith("verdachte is veroordeeld tot 48 maanden")
    assert u.url.endswith("ECLI:NL:RBNNE:2024:674")


def test_rechtbank_sections_and_paragraphs_are_traceable():
    u = parse_xml((FIXTURES / "ecli_rbnne_2024_674.xml").read_bytes())
    titles = [s.titel for s in u.secties]
    assert titles[0] == "Inhoudsindicatie"
    assert "Beoordeling van het bewijs › Oordeel van de rechtbank" in titles
    bewijs = next(s for s in u.secties if s.titel.endswith("› Oordeel van de rechtbank"))
    assert bewijs.paragrafen[0].nummer == 1
    assert [p.nummer for p in bewijs.paragrafen] == list(range(1, len(bewijs.paragrafen) + 1))
    numbered = [p for p in bewijs.paragrafen if p.label]
    assert numbered and numbered[0].label == "1"
    assert all(p.tekst.strip() for s in u.secties for p in s.paragrafen)
    # list items and table cells become paragraphs too, nothing is silently dropped
    bewijs_all = [p for s in u.secties if s.titel.startswith("Beoordeling") for p in s.paragrafen]
    assert any("Fors/stevig postuur" in p.tekst for p in bewijs_all)
    assert any("AAPQ3474NL#03" in p.tekst for p in bewijs_all)
    assert [s.nummer for s in u.secties][:3] == ["0", "1", "1.1"]


def test_hof_ruling_parses_with_different_layout():
    u = parse_xml((FIXTURES / "ecli_gharl_2024_2.xml").read_bytes())
    assert u.ecli == "ECLI:NL:GHARL:2024:2"
    assert u.instantie_type == "Gerechtshof"
    titles = [s.titel for s in u.secties]
    assert "Het hoger beroep" in titles
    assert "BESLISSING" in titles
    beslissing = next(s for s in u.secties if s.titel == "BESLISSING")
    assert beslissing.paragrafen[0].tekst == "Het hof:"
    # text before the first <section> is kept as its own "Kop" section
    kop = next(s for s in u.secties if s.titel == "Kop")
    assert any("Parketnummer" in p.tekst for p in kop.paragrafen)


def test_nested_sections_get_hierarchical_numbers():
    xml = b"""<open-rechtspraak>
      <rdf:RDF xmlns:rdf="r" xmlns:dcterms="d"><rdf:Description>
        <dcterms:identifier>ECLI:NL:HR:2020:1</dcterms:identifier>
        <dcterms:creator>Hoge Raad</dcterms:creator>
        <dcterms:date>2020-01-02</dcterms:date>
      </rdf:Description></rdf:RDF>
      <uitspraak xmlns="http://www.rechtspraak.nl/schema/rechtspraak-1.0">
        <section><title><nr>3</nr> Beoordeling</title><para>intro</para>
          <section><title>3.1 Eerste middel</title><para>a</para><para>b</para></section>
          <section><title>Tweede middel</title><para>c</para></section>
        </section>
      </uitspraak></open-rechtspraak>"""
    u = parse_xml(xml)
    assert u.instantie_type == "Hoge Raad"
    assert [(s.nummer, s.titel) for s in u.secties] == [
        ("3", "Beoordeling"),
        ("3.1", "Eerste middel"),
        ("3.2", "Tweede middel"),
    ]


def test_lone_bold_paragraphs_become_subsection_titles():
    xml = b"""<open-rechtspraak>
      <rdf:RDF xmlns:rdf="r" xmlns:dcterms="d"><rdf:Description>
        <dcterms:identifier>ECLI:NL:RBAMS:2024:1</dcterms:identifier>
        <dcterms:creator>Rechtbank Amsterdam</dcterms:creator>
        <dcterms:date>2024-03-01</dcterms:date>
      </rdf:Description></rdf:RDF>
      <uitspraak>
        <section><title>4 Waardering van het bewijs</title>
          <para>Inleidende alinea.</para>
          <para><emphasis role="bold">Het standpunt van de verdediging</emphasis></para>
          <para>De raadsman heeft vrijspraak bepleit.</para>
          <para><emphasis role="underline">Het oordeel van de rechtbank</emphasis></para>
          <para>De rechtbank acht het feit bewezen.</para>
          <para><emphasis role="bold">en</emphasis></para>
          <para>Deze losse 'en' is geen kop.</para>
          <para><emphasis role="italic">Cursief is geen kop</emphasis></para>
          <para><emphasis role="bold">[verdachte] ,</emphasis></para>
        </section>
      </uitspraak></open-rechtspraak>"""
    u = parse_xml(xml)
    assert [(s.nummer, s.titel, len(s.paragrafen)) for s in u.secties] == [
        ("4", "Waardering van het bewijs", 1),
        ("4.1", "Waardering van het bewijs › Het standpunt van de verdediging", 1),
        ("4.2", "Waardering van het bewijs › Het oordeel van de rechtbank", 5),
    ]
    assert u.secties[2].paragrafen[0].nummer == 1
    assert u.secties[2].paragrafen[1].tekst == "en"


def test_document_without_sections_is_one_section_named_uitspraak():
    xml = b"""<open-rechtspraak>
      <rdf:RDF xmlns:rdf="r" xmlns:dcterms="d"><rdf:Description>
        <dcterms:identifier>ECLI:NL:RBAMS:2024:2</dcterms:identifier>
        <dcterms:creator>Rechtbank Amsterdam</dcterms:creator>
        <dcterms:date>2024-03-01</dcterms:date>
      </rdf:Description></rdf:RDF>
      <uitspraak><para>Alleen losse tekst.</para><para>Zonder secties.</para></uitspraak>
      </open-rechtspraak>"""
    u = parse_xml(xml)
    assert [(s.nummer, s.titel, len(s.paragrafen)) for s in u.secties] == [("1", "Uitspraak", 2)]


def test_roundtrip_json(tmp_path):
    u = parse_xml((FIXTURES / "ecli_rbnne_2024_674.xml").read_bytes())
    path = u.save(tmp_path)
    again = Uitspraak.from_json(path.read_text(encoding="utf-8"))
    assert again == u


def test_broken_document_yields_error_record(tmp_path):
    (tmp_path / "a.xml").write_bytes(b"<open-rechtspraak><nope/></open-rechtspraak>")
    (tmp_path / "b.xml").write_bytes(b"<not xml")
    (tmp_path / "c.xml").write_bytes((FIXTURES / "ecli_gharl_2024_2.xml").read_bytes())
    results = list(parse_dir(tmp_path))
    assert isinstance(results[0], ParseError) and "Description" in results[0].fout
    assert isinstance(results[1], ParseError) and results[1].bron == "b.xml"
    assert isinstance(results[2], Uitspraak)
    assert isinstance(parse_file(tmp_path / "b.xml"), ParseError)


def test_long_text_before_first_section_is_body_not_header():
    paras = "".join(f"<para>Alinea {i} van de uitspraak.</para>" for i in range(20))
    xml = f"""<open-rechtspraak>
      <rdf:RDF xmlns:rdf="r" xmlns:dcterms="d"><rdf:Description>
        <dcterms:identifier>ECLI:NL:RBAMS:2024:3</dcterms:identifier>
        <dcterms:creator>Rechtbank Amsterdam</dcterms:creator>
        <dcterms:date>2024-03-01</dcterms:date>
      </rdf:Description></rdf:RDF>
      <uitspraak>{paras}<section><title>Beslissing</title><para>Vrijspraak.</para></section>
      </uitspraak></open-rechtspraak>""".encode()
    u = parse_xml(xml)
    assert [(s.nummer, s.titel) for s in u.secties] == [("1", "Uitspraak"), ("2", "Beslissing")]
