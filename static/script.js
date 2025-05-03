const matrix = document.getElementById('matrix');
const mCtx   = matrix.getContext('2d');

const confCanvas = document.createElement('canvas');
confCanvas.id = 'confetti-canvas';
document.body.appendChild(confCanvas);
const cCtx = confCanvas.getContext('2d');

function resizeAll(){
  matrix.width = confCanvas.width = window.innerWidth;
  matrix.height= confCanvas.height= window.innerHeight;
}
window.addEventListener('resize', resizeAll);
resizeAll();

const fontSize = 16;
let columns  = Math.floor(matrix.width / fontSize);
let drops    = Array(columns).fill(0);
const chars  = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%^&*'.split('');

function drawMatrix(){
  mCtx.fillStyle = 'rgba(0,0,0,0.05)';
  mCtx.fillRect(0, 0, matrix.width, matrix.height);
  mCtx.fillStyle = '#f00';
  mCtx.font = `${fontSize}px monospace`;
  drops.forEach((y,i)=>{
    let text = chars[Math.floor(Math.random()*chars.length)];
    let x = i * fontSize;
    mCtx.fillText(text, x, y * fontSize);
    if (y * fontSize > matrix.height || Math.random() > 0.975) drops[i] = 0;
    else drops[i]++;
  });
}

let confettiParticles = [];
class Confetti {
  constructor(){
    this.x     = Math.random()*confCanvas.width;
    this.y     = -10;
    this.r     = Math.random()*6 + 2;
    this.d     = Math.random()*confCanvas.height/2;
    this.color = `hsl(${Math.random()*360},100%,50%)`;
    this.tilt  = Math.random()*10-10;
  }
  draw(){
    cCtx.beginPath();
    cCtx.lineWidth = this.r;
    cCtx.strokeStyle = this.color;
    cCtx.moveTo(this.x + this.tilt + this.r/2, this.y);
    cCtx.lineTo(this.x + this.tilt, this.y + this.tilt + this.r/2);
    cCtx.stroke();
  }
  update(){
    this.y += (Math.cos(0.01*this.d) + 3 + this.r/2)/2;
    this.x += Math.sin(0.01*this.d);
    this.tilt = Math.sin(0.05*this.d) * 15;
    if (this.y > confCanvas.height) {
      this.y = -20;
      this.x = Math.random()*confCanvas.width;
    }
  }
}

function launchConfetti(){
  confettiParticles = [];
  for(let i=0;i<200;i++) confettiParticles.push(new Confetti());
  requestAnimationFrame(confettiLoop);
}
function confettiLoop(){
  cCtx.clearRect(0,0,confCanvas.width, confCanvas.height);
  confettiParticles.forEach(p=>{ p.draw(); p.update(); });
  requestAnimationFrame(confettiLoop);
}

function initEffects(){
  const card = document.querySelector('.result-card');
  if (!card) return;
  if (card.classList.contains('harmful')){
    document.body.classList.add('hacked');
    setInterval(drawMatrix, 50);
  } else {
    launchConfetti();
  }
}

document.addEventListener('DOMContentLoaded', initEffects);

