from pathlib import Path
from xml.etree import ElementTree as ET
MODEL=Path(__file__).resolve().parents[1]/'models'/'humanoid1.xml'

def test_xml_structure():
    root=ET.parse(MODEL).getroot()
    joints={j.attrib['name'] for j in root.findall('.//worldbody//joint') if 'name' in j.attrib}
    motors=root.findall('.//actuator/motor')
    assert len(motors)==16, len(motors)
    assert all(m.attrib['joint'] in joints for m in motors)
    assert len({m.attrib['joint'] for m in motors})==16
    assert len(root.findall('.//freejoint'))==1

if __name__=='__main__':
    test_xml_structure();print('PASS: XML parses, 16 actuator-joint mappings, one free root')
