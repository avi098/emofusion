// worklet-processor.js
class AudioLevelProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._lastUpdate = currentTime;
    this._updateIntervalInMS = 100;
    this.volume = 0;
    this.clipCount = 0;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];

    if (!input || !input.length) return true;

    const samples = input[0];
    let sum = 0;
    let clipCount = 0;

    // Calculate average volume and count clips
    for (let i = 0; i < samples.length; ++i) {
      const absolute = Math.abs(samples[i]);
      sum += absolute;
      if (absolute > 0.95) {
        clipCount++;
      }
    }

    // Update volume and clip count
    this.volume = Math.sqrt(sum / samples.length);
    this.clipCount = clipCount;

    // Send message to main thread if enough time has passed
    const currentTime = globalThis.currentTime;
    if (currentTime - this._lastUpdate > this._updateIntervalInMS / 1000) {
      this.port.postMessage({
        volume: this.volume,
        clipCount: this.clipCount,
        time: currentTime,
      });
      this._lastUpdate = currentTime;
    }

    return true;
  }
}

registerProcessor("audio-level-processor", AudioLevelProcessor);
