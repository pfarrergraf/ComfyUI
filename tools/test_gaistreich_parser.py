import runpy
import json
import traceback

PATH = "custom_nodes/gaistreich-batch-prompt/batch_prompt.py"

# The ACE batch JSON provided by the user
INPUT_JSON = r'''
[
  {
    "id": "obros_style_rap_001",
    "tags": "Christian street rap, energetic, youth, 94 bpm, 4/4, F minor",
    "lyrics": "[Hook]\nIch fall nicht tief, ich fall nach oben,\nauch wenn die Stra\u00dfen meine Fehler loben.\nKein Gold, kein Hype, kein falscher Thron -\nmein Wert kommt nicht vom Applaus der Crowd.\n\n[Part 1]\nHab N\u00e4chte gez\u00e4hlt mit Kopfh\u00f6rern laut,\nHab Tr\u00e4ume gebaut und wieder zerhau'n.\nZwischen Insta-Filtern und Leistungsdruck\ngef\u00fchlt sich mein Herz manchmal eingedr\u00fcckt.\n\nFreunde sagen: \u201eBruder, mach Cash!\u201c\nDoch Cash heilt keinen Seelen-Crash.\nHab gelernt: Erfolg ist laut,\naber Wahrheit wird leise gebaut.\n\n[Spoken]\nNicht jeder Sieg ist ein Gewinn.\nNicht jeder Verlust ist das Ende.\n\n[Hook]\nIch fall nicht tief, ich fall nach oben,\nweil Gnade st\u00e4rker ist als mein Toben.\nKein Fame definiert meinen Stand -\nich steh, weil Gott mich h\u00e4lt in der Hand.",
    "bpm": 94,
    "duration": 185,
    "timesignature": "4",
    "language": "de",
    "keyscale": "F minor"
  },
  {
    "id": "obros_style_rap_002",
    "tags": "Christian motivational rap, hard drums, youth energy, 98 bpm, 4/4, E minor",
    "lyrics": "[Hook]\nIch renn durch Feuer, doch brenn nicht aus,\nkein Schatten l\u00f6scht mein inneres Haus.\nWas auch passiert, ich bleib stabil,\nweil meine Hoffnung h\u00f6her zielt.\n\n[Part 1]\n16 und Druck von allen Seiten,\nNoten, Zukunft, Erwartungen leiten.\nDu willst performen, willst stark erschein\u2019n,\naber f\u00fchlst dich heimlich viel zu klein.\n\nJeder redet von Level-Up,\naber keiner hebt dich wirklich auf.\nIch hab gelernt in dunkler Nacht:\nNicht mein Mut hat mich stark gemacht.\n\n[Spoken]\nWenn keiner klatscht,\nbleibt trotzdem Wert.\n\n[Hook]\nIch renn durch Feuer, doch brenn nicht aus,\nweil Gott mein Fundament gebaut.\nNicht Likes, nicht Ruhm, nicht Applaus -\nSeine Treue h\u00e4lt mich raus.",
    "bpm": 98,
    "duration": 190,
    "timesignature": "4",
    "language": "de",
    "keyscale": "E minor"
  },
  {
    "id": "obros_style_rap_003",
    "tags": "Christian trap rap, modern youth vibe, 100 bpm, 4/4, G minor",
    "lyrics": "[Hook]\nKein Fake, kein Spiel, kein doppeltes Gesicht,\nmein Leben geh\u00f6rt nicht dem Rampenlicht.\nWenn ich fall, steh ich wieder auf,\nweil Liebe st\u00e4rker ist als mein Lauf.\n\n[Part 1]\nZwischen Schule, Stress und Selbstzweifeln,\ngef\u00fchlt sich alles an wie Kreislaufen.\nDu willst raus aus dem grauen Beton,\nwillst wissen: Wof\u00fcr leb ich schon?\n\nFreunde wechseln wie Trends im Feed,\nTreue ist selten wie echter Frieden.\nDoch ich hab mehr als nur meinen Plan,\nich wei\u00df, da ist Einer, der tr\u00e4gt mich voran.\n\n[Spoken]\nNicht perfekt.\nNicht heilig.\nAber gehalten.\n\n[Hook]\nKein Fake, kein Spiel, kein doppeltes Gesicht,\nmein Wert kommt nicht vom Rampenlicht.\nWenn ich fall, steh ich wieder auf -\nweil Gnade mich hebt in meinem Lauf.",
    "bpm": 100,
    "duration": 180,
    "timesignature": "4",
    "language": "de",
    "keyscale": "G minor"
  },
  {
    "id": "obros_style_rap_004",
    "tags": "Christian hype rap, powerful youth anthem, 92 bpm, 4/4, D minor",
    "lyrics": "[Hook]\nIch bleib echt, auch wenn die Welt sich dreht,\nbleib fest, wenn der Boden vergeht.\nKein Sturm nimmt mir meine Sicht,\nmein K\u00f6nig steht hinter mir - ich nicht.\n\n[Part 1]\nHab Fehler gemacht, hab Chancen vertan,\nmanchmal war ich mein gr\u00f6\u00dfter Feind sogar.\nDoch jedes Mal, wenn alles zerbricht,\nsp\u00fcr ich: Ich k\u00e4mpf hier nicht allein im Licht.\n\nSie sagen: \u201eMach schneller, mach mehr!\u201c\nDoch meine Seele wurde schwer.\nHab gemerkt, ich brauch keinen Thron,\nnur festen Halt auf Gottes Sohn.\n\n[Hook]\nIch bleib echt, auch wenn die Welt sich dreht,\nbleib fest, wenn der Boden vergeht,\nKein Sturm nimmt mir meine Sicht,\nmein K\u00f6nig steht hinter mir - ich nicht.",
    "bpm": 92,
    "duration": 195,
    "timesignature": "4",
    "language": "de",
    "keyscale": "D minor"
  },
  {
    "id": "obros_style_rap_005",
    "tags": "Christian street anthem, emotional but strong, 96 bpm, 4/4, A minor",
    "lyrics": "[Hook]\n Ich bin mehr als mein Fehler, mehr als mein Name,\nmehr als mein Ruf in dieser Stra\u00dfe.\nWenn alles wankt, bleib ich steh\u2019n,\nweil Seine Hand mich l\u00e4sst nicht geh\u2019n.\n\n[Part 1]\nIch hab gezweifelt, hab viel versteckt,\nmein Herz war laut, nur mein Mund perfekt.\nDoch nachts, wenn alles stiller wird,\nmerk ich, wie Seine Wahrheit mich ber\u00fchrt.\n\nNicht jede Tr\u00e4ne ist verlor\u2019n,\nnicht jeder Kampf ist umsonst gebor\u2019n.\nManchmal wächst Mut im tiefsten Fall,\nund Gnade bricht durch Beton und Stahl.\n\n[Hook]\nMehr als mein Fehler, mehr als mein Name,\nmein Wert steht fest - trotz meiner Narben.\nWenn alles wankt, bleib ich steh\u2019n,\nweil Seine Liebe mich l\u00e4sst nicht geh\u2019n.",
    "bpm": 96,
    "duration": 185,
    "timesignature": "4",
    "language": "de",
    "keyscale": "A minor"
  }
]
'''


def main():
    try:
        mod = runpy.run_path(PATH)
        # Retrieve iterator class
        Iterator = mod.get('BATCH_audio_ace_iterator')
        if Iterator is None:
            print(json.dumps({'error': 'BATCH_audio_ace_iterator not found in module'}, ensure_ascii=False))
            return

        it = Iterator()
        # Call process on the full JSON with index 0
        result = it.process(INPUT_JSON, 'json', 0)
        # Convert to JSON-friendly structure
        print(json.dumps({'result': result}, ensure_ascii=False))
    except Exception:
        tb = traceback.format_exc()
        print(json.dumps({'exception': tb}, ensure_ascii=False))


if __name__ == '__main__':
    main()
