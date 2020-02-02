import os
import sys
sys.path.append('waveglow/')
os.environ['PYTHONASYNCIODEBUG'] = '1'

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import torch
    import numpy as np

    import asyncio, socket

    from hparams import create_hparams
    from model import Tacotron2
    from layers import TacotronSTFT, STFT
    from audio_processing import griffin_lim
    from text import text_to_sequence
    #from denoiser import Denoiser


class TTS():
    def __init__(self, hparams, tacotron2_path, waveglow_path):
        self.Tacotron2Model = Tacotron2(hparams).cpu()
        self.WaveglowModel = None
        self.Denoiser = None

        self.WaveglowSigma = 0.800
        self.UseDenoiser = False
        self.DenoiserStrength = 0.01
        
        self.load_models(tacotron2_path, waveglow_path)

    def load_models(self, tacotron2_path, waveglow_path):
        if tacotron2_path is not None:
            state_dict = torch.load(tacotron2_path, map_location='cpu')['state_dict']
            self.Tacotron2Model.load_state_dict(state_dict)
            self.Tacotron2Model.cpu().eval()

        if waveglow_path is not None:
            self.WaveglowModel = torch.load(waveglow_path, map_location='cpu')['model'] 
            self.WaveglowModel.cpu().eval()
            #self.Denoiser = Denoiser(self.WaveglowModel)

    def inference(self, text):
        text += ';.'
        sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cpu().long()

        _, mel_outputs_postnet, _, _ = self.Tacotron2Model.inference(sequence)

        with torch.no_grad():
            audio = self.WaveglowModel.infer(mel_outputs_postnet, self.WaveglowSigma)

        #if(self.UseDenoiser == True)
        #    audio = self.Denoiser(audio, self.DenoiserStrength)[:, 0]
        audio = audio[0].cpu().numpy()
        audio *= 32767 / np.max(np.abs(audio))
        # convert to 16-bit data
        audio = audio.astype(np.int16)

        return audio


class TTSServer():
    def __init__(self, hparams, tacotron2_path, waveglow_path):
        self.TTS = TTS(hparams, tacotron2_path, waveglow_path)
        self.EventLoop = None
        #self.logger = logging.getLogger('Server')


    #async def run_server(host, port):
    #    server = await asyncio.start_server(serve_client, host, port)
    #    await server.serve_forever()

    def start_server(self, ip, port):
        self.EventLoop = asyncio.get_event_loop()
        self.EventLoop.create_task(asyncio.start_server(self.serve_client, ip, port))
        self.EventLoop.run_forever()

    async def serve_client(self, reader, writer):
        print('Client connected')

        while True:
            try:
                data = await reader.read(512)
                if not data:
                    break
                message = data.decode('utf8')
                print('Message received: ' + message)

                audio = self.TTS.inference(message).tobytes()
                length = len(audio)
                print(length)    

                writer.write(length.to_bytes(4, byteorder='little') + audio)
                await writer.drain()
                #await write_response(writer, response, cid)
                print('Message sent')

            except ConnectionResetError: 
                print('Exception')
                break
        
        print('END OF LOOP')
        writer.close()
        self.EventLoop.stop()


def main():
    hparams = create_hparams()
    hparams.sampling_rate = 22050

    ttsServer = TTSServer(hparams, 'checkpoint_7000', 'waveglow_256channels_ljs_v2.pt')

    ttsServer.start_server('DESKTOP-4J6T4O5', 17853)

    print('End')


if __name__ == '__main__':
    main()
