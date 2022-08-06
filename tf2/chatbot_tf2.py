import json
import os
import numpy as np
import tensorflow as tf
import pyttsx3
from pyttsx3 import engine
import speech_recognition as sr
from tf2 import encoder
from tf2 import model
from tf2 import sample

engine = pyttsx3.init()
""" RATE"""
rate = engine.getProperty('rate')
engine.setProperty('rate', 190)

"""VOLUME"""
volume = engine.getProperty('volume')
engine.setProperty('volume',1.0)

"""VOICE"""
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

def interact_model(
    temperature,
    top_k,
    top_p,
    nsamples,
    batch_size,
    length,
    seed=None,
):

    models_dir = os.path.expanduser(os.path.expandvars('./models'))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder('345M_org', models_dir)
    hparams = model.default_hparams()
    with open(os.path.join('./models', '345M_org', 'hparams.json')) as f:
        hparams.update(json.load(f))

    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        contxt = tf.compat.v1.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=contxt,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.compat.v1.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('./models', '345M_org'))
        saver.restore(sess, ckpt)

        raw_text = '<|endofdlg|>'
        print('#' * 20 + ' Start ' + '#' * 20)
        while True:
            with sr.Microphone() as source:
                r = sr.Recognizer()
                r.pause_threshold=1
                audio=r.listen(source)
                input_utt = r.recognize_google(audio)
                raw_text +='\n' + 'You: '+ input_utt + '\n' + 'Jarvis: '
                contxt_tokens = enc.encode(raw_text)
                print('Recognizing:')
                for _ in range(nsamples // batch_size):
                    out = sess.run(output, feed_dict={
                    contxt: [contxt_tokens for _ in range(batch_size)]
                })[:, len(contxt_tokens):]
                
                    for i in range(batch_size):
                        text = enc.decode(out[i])
                        result=list(text.partition('\n'))
                        print('You said:- ' + r.recognize_google(audio))
                        print('Jarvis:' + result[0])
                        raw_text += str(result[0])
                        engine.say(result[0])
                        engine.runAndWait()
                        True
                        if sr.UnknownValueError:
                            print('Speak Now:')
                            engine.runAndWait
                            True
            try:
                engine.runAndWait
                True
            except sr.UnknownValueError:
                engine.runAndWait
                True
    
