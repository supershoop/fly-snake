import { useEffect, useState } from 'react';
import { QRCodeSVG } from 'qrcode.react';
import './AudienceTraining.css';

export function AudienceTraining({ urls }: { urls: string[] }) {
  // A tunnel can be added without restarting the brain or losing its live learner.
  const publicUrl = import.meta.env.VITE_FEEDBACK_URL?.trim();
  const links = publicUrl ? [publicUrl] : urls;
  const [choice, setChoice] = useState('');
  const [copied, setCopied] = useState('');
  const url = links.includes(choice) ? choice : links[0];
  const internet = Boolean(url?.startsWith('https://'));
  useEffect(() => { setCopied(''); }, [url]);
  const copy = async () => {
    try { await navigator.clipboard.writeText(url); setCopied('Link copied.'); }
    catch { setCopied('Copy the link shown above to share it.'); }
  };

  return <section className="audience-training" aria-labelledby="audience-title">
    <div className="audience-copy">
      <strong id="audience-title">TRAIN FROM YOUR PHONE</strong>
      <h3>Scan. Watch. Steer.</h3>
      <p>Join this live experiment and press the way the fly should have moved. Everyone shares one unit of influence per move, so a big crowd guides the readout without overruling what the fly learns from food and collisions.</p>
      <p>{internet ? 'Stay on eduroam, any Wi-Fi, or mobile data and scan the QR code.' : 'Connect to the demo’s Wi-Fi, then scan the QR code.'} The phone page shows the fly’s moves and confirms each direction taught.</p>
      {url ? <>
        {links.length > 1 && <label>Demo network
          <select aria-label="Demo network address" value={url} onChange={event => setChoice(event.target.value)}>
            {links.map(value => <option key={value} value={value}>{new URL(value).host}</option>)}
          </select>
        </label>}
        <a className="audience-link" href={url} target="_blank" rel="noreferrer">{url}</a>
        <div className="controls"><button onClick={copy}>Copy phone link</button><a href={url} target="_blank" rel="noreferrer">Open controller ↗</a></div>
        <p className="audience-copy-result" role="status">{copied}</p>
        <details><summary>Phone can’t connect?</summary>{internet
          ? <p>The demo laptop and its sharing connection must stay running. If sharing was restarted, refresh this page and scan the new code.</p>
          : <p>Use the same Wi-Fi or hotspot as the demo. Campus Wi-Fi may block connections between devices; use an HTTPS sharing link in that case. The brain server must allow network connections (<code>--host 0.0.0.0</code>). If several addresses appear, choose the demo’s Wi-Fi address.</p>}</details>
      </> : <p role="status">Restart the updated brain server with <code>--host 0.0.0.0</code> to get a phone link.</p>}
    </div>
    {url && <a className="audience-qr" href={url} target="_blank" rel="noreferrer" aria-label="Open the phone feedback controller">
      <QRCodeSVG value={url} size={240} level="M" marginSize={4} title="Scan to teach the fly which way to move"/>
      <span>Scan to train the fly</span>
    </a>}
  </section>;
}
