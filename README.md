# Trabajo Práctico Especial

<h2 class="pm-node nj-subtitle">Demodulación de tonos de discado de teléfono</h2>

# Introducción

El sistema de discado por tonos de nuestros teléfonos utiliza los principios de la codificación DTMF, o Dual Tone Multi Frequency. Este sistema de codificación convierte los códigos de información (los 10 dígitos decimales en el caso del discado) en otras tantas señales analógicas, cuya frecuencia debe estar contenida en el rango de las frecuencias de la voz humana, o más bien en el rango de frecuencias del canal telefónico. De este modo se crea un sistema de codificación que puede ser interpretado por todos los sistemas que se conectan a la red telefónica, como (obviamente) teléfonos, pero también, modems, máquinas de fax, centrales telefónicas, hubs, switches, y hasta las distribuidoras de televisión por cable.

La versión de DTMF utilizada en el discado delefónico se conoce con la marca registrada "Touch-Tone" y está estandarizada por la recomendsación Q.23 de la ITU-T. El código es muy sencillo y consiste en combinar dos tonos de distintas frecuencias (Dual Tone), de un número de frecuencias distintas (Multi Frequency) pero conocidaas y fijas elegidas según se indica en la tabla siguiente:

# Código auxiliar

```julia id=4bf5db09-fe71-41c7-8154-d3a5d04ef871
# Cargo paquetitos
using DSP, FFTW, Statistics, WAV, Base, DataFrames


function wavread_mono(file)
  x, sr = wavread(file)
  return mean(x; dims=2)[:], sr
end

# Y armo un par de funciones auxiliares
stem(args...; kwargs...) = sticks(args...; 
  																marker=:circle, 
  																leg=false, 
  																kwargs...)

zeropolegain(pr) = DSP.ZeroPoleGain(pr)
zeropolegain(z, p, g) = DSP.ZeroPoleGain(z, p, g)
polynomialratio(zpg) = DSP.PolynomialRatio(zpg)
function polynomialratio(b, a)
  n = max(length(a), length(b))
  return DSP.PolynomialRatio(padright(b, n), padright(a, n))
end
getpoles(zpg) = DSP.ZeroPoleGain(zpg).p
getzeros(zpg) = DSP.ZeroPoleGain(zpg).z
getgain(zpg) = DSP.ZeroPoleGain(zpg).k
getnumcoefs(pr) = trimlastzeros!(reverse(DSP.PolynomialRatio(pr).b.coeffs))
getdencoefs(pr) = trimlastzeros!(reverse(DSP.PolynomialRatio(pr).a.coeffs))
function trimlastzeros!(a)
  !iszero(a[end]) && return a
  pop!(a)
  return trimlastzeros!(a)
end

DSP.filt(zpg::DSP.ZeroPoleGain, r...; kwargs...) = filt(polynomialratio(zpg), r...; kwargs...)

function zplane(zs, ps; kwargs...)
	scatter(real.(zs), imag.(zs);
		  marker = (:black, :circle), label="Cero", kwargs...)
	scatter!( real.(ps), imag.(ps);
	  	marker = (:red, :xcross), label="Polo", kwargs...)
end

function zplane(zpg; kwargs...)
	zs = getzeros(zpg)
	ps = getpoles(zpg)
	isempty(zs) && isempty(ps) && (append!(ps, 0))
	
	return zplane(zs, ps; kwargs...)
end
zplane(pr::DSP.PolynomialRatio; kwargs...) = zplane(DSP.ZeroPoleGain(pr); kwargs...)



# Delta
d(n) = n == 0 ? 1. : 0. 

# Escalón
u(n) = n >= 0 ? 1. : 0. 

using Plots
Plots.default(:legend, false)

# Pad vector with zeros on the right until its length is `n`
padright(x, n) = copyto!(zeros(eltype(x), n), x)

"""
Función módulo pero con offset (opcional)
Manda a `t` al intervalo [from, from+length)
sumándole o restándole algún múltiplo de `len`
"""
cshift(t, len; from=0) = mod(t - from, len) + from

# Espectrograma
using IterTools
function stft(x; overlap, window, nfft, rest...)
  nwin = length(window)
  @assert overlap < nwin

  res = [ fft(padright(xseg .* window, nfft)) 
    for xseg in partition(x, nwin, nwin - overlap)]
  
  return [ res[i][j] for j in 1:nfft, i in eachindex(res)]
end

function specgram(x; 
    overlap=0.5, 
    window=hamming(div(length(x), 16)), 
    nfft=length(window), 
    rest...)
  
  window isa Integer && (window = rect(window))
  overlap isa AbstractFloat && (overlap = round(Int, length(window) * overlap))
    
  return stft(x; overlap=overlap, window=window, nfft=nfft)
end
  
specplot(x::AbstractMatrix; kwargs...) = @error "You are entering a Matrix (2D Array). I need a Vector (1D Array)."
function specplot(x::AbstractVector; 
      fs=1, 
      onesided=false, 
      xaxis="Tiempo (s)", 
      yaxis="Frecuencia (Hz)",
      kws...)
    mat = specgram(x; kws...)
  
    fmax = fs
    if onesided
      mat = mat[1:div(size(mat, 1) + 2, 2), :]
      fmax = fs/2
    end
  
  times = range(0; length=size(mat, 2), stop=length(x)/fs) # aprox
  freqs = range(0; length=size(mat, 1), stop=fmax)
  
	# Reubico las frecuencias negativas arriba de todo
  if !onesided
    freqs = cshift.(freqs, fs, -fs/2)
    ord   = sortperm(freqs)
    mat   = mat[ord, :]
    freqs = freqs[ord]
  end

	return heatmap(times, freqs, log.(abs.(mat) .+ eps()); 
          xaxis=xaxis, yaxis=yaxis,
          seriescolor=:bluesreds, legend=true, kws...)
 return times, freqs, mat 
end
function specplot(x :: AbstractVector{<:AbstractFloat}; kws...)
    return specplot(convert.(Complex, x); onesided=true, kws...)
end

```

# Desarrollo

![screenshot.png][nextjournal#file#426d1501-3a13-436b-8626-baaa90743947]

La duración de cada señal para ser considerada un dígito es variable. Cada dígito debe tener una longitud de 70 ms como mínimo, aunque algunos equipos pueden aceptar menor duración de pulso

Utilice la señal *`dtmf2.wav`*. La señal contiene un código DTMF (convertido a señal discreta mediante un A/D). 

[dtmf2.wav][nextjournal#file#fa3fd099-8ada-43ea-a176-770134676358]

```julia id=575c5f5b-2957-4b1c-8683-001d4a8bc044
x, sr = wavread_mono( [reference][nextjournal#reference#bb4c58bd-1bf1-406a-a857-e0bd58b42c39]);
```

```julia id=2e2671f2-1b8d-4f1d-aecc-545d1271c929
#= 
Escribiendo el archivo en la carpeta /results/ lo obtienen como salida.
=#
wavwrite(x, "/results/out.wav"; Fs=sr)
```

[out.wav][nextjournal#output#2e2671f2-1b8d-4f1d-aecc-545d1271c929#out.wav]

## 1. Caracterización temporal

```julia id=00f9cce1-407f-41f0-a89a-5df7b192c0c5
time_sample=(0:(length(x)-1))/sr;
plot(time_sample,x)
```

![result][nextjournal#output#00f9cce1-407f-41f0-a89a-5df7b192c0c5#result]

```julia id=6110fad1-f177-4d7e-ae38-da56153e339e
# a vector holding the 6 vectors of samples
tsegs = [ (0.215, 0.35), (0.58, 0.66), (1.01, 1.13), (1.45, 1.56), (1.86, 1.98), (2.31, 2.45) ]
nsegs = [ clamp.(round.(Int, sr .* ts), 1, length(x)) for ts in tsegs ]
xsegs = [ x[n0:ne] for (n0, ne) in nsegs ]

# a vector holding the 6 vectors of indices
nss   = [ n0:ne for (n0, ne) in nsegs ]

# a vector holding the 6 vectors of time stamps
tss   = nss ./ sr

# make 4 plots
pls = [ 
        plot(ts, xseg;
    legend=true, label=string(ts[1]) * " s to " * string(ts[end]) * " s"
      ) 
      for (ts, xseg) in zip(tss, xsegs) ]

# and plot them in a 3 by 2 matrix
plot(pls...; layout = (3, 2), label=["Tono 1" "Tono 2" "Tono 3" "Tono 4" "Tono 5" "Tono 6"])
```

![result][nextjournal#output#6110fad1-f177-4d7e-ae38-da56153e339e#result]

```julia id=827fc4fe-e33d-45c0-acb4-c0bbdee00cfd
time_sample=(0:(length(x)-1))/sr;
plot(time_sample,x,xticks =0.215:0.01:0.35; xlims=(0.215, 0.35),title="Tono 1",xlabel="Tiempo [s]",ylabel="x(t)")
```

![result][nextjournal#output#827fc4fe-e33d-45c0-acb4-c0bbdee00cfd#result]

```julia id=25028c9c-3a6b-4623-875a-bc484beaaa57
time_sample=(0:(length(x)-1))/sr;
plot(time_sample,x,xticks =0.58:0.01:0.66)
plot!(; xlims=(0.58, 0.66),title="Tono 2",xlabel="Tiempo [s]",ylabel="x(t)")
```

![result][nextjournal#output#25028c9c-3a6b-4623-875a-bc484beaaa57#result]

```julia id=b8f245ea-096c-4994-9e53-2f90ace5dcdb
time_sample=(0:(length(x)-1))/sr;
plot(time_sample,x,xticks =1.01:0.01:1.13; xlims=(1.01, 1.13),title="Tono 3",xlabel="Tiempo [s]",ylabel="x(t)")
```

![result][nextjournal#output#b8f245ea-096c-4994-9e53-2f90ace5dcdb#result]

```julia id=e1c3e588-a573-42a7-b123-606ad7686ece
time_sample=(0:(length(x)-1))/sr;
plot(time_sample,x,xticks =1.45:0.01:1.56)
plot!(; xlims=(1.45, 1.56),title="Tono 4",xlabel="Tiempo [s]",ylabel="x(t)")
```

![result][nextjournal#output#e1c3e588-a573-42a7-b123-606ad7686ece#result]

```julia id=77e4bc62-63ab-413d-ba11-1c8f86326967
time_sample=(0:(length(x)-1))/sr;
plot(time_sample,x,xticks =1.86:0.01:1.98; xlims=(1.86, 1.98),title="Tono 5",xlabel="Tiempo [s]",ylabel="x(t)")
```

![result][nextjournal#output#77e4bc62-63ab-413d-ba11-1c8f86326967#result]

```julia id=49d678c4-bc8e-425b-829f-263bc17bf9da
time_sample=(0:(length(x)-1))/sr;
plot(time_sample,x,xticks =2.31:0.01:2.45; xlims=(2.31, 2.45),title="Tono 6",xlabel="Tiempo [s]",ylabel="x(t)")
```

![result][nextjournal#output#49d678c4-bc8e-425b-829f-263bc17bf9da#result]

En las siguientes tablas se pasa a mostrar el principio y el fin de cada tono y luego la duración (aproximada).

![ppiofin.jpg][nextjournal#file#90d69ca0-3e5e-4430-8c1c-e56f9878b591]

![duracion.jpg][nextjournal#file#1d662761-2f85-493c-bb8b-b5b834d22e49]

Los tiempos entre principio y fin de cada tono corresponde a ruido y silencios.

#### ¿Podría considerarse que las señales del código DTMF corresponden a una sección temporal de una señal periódica infinita? Si es así, identifique la frecuencia fundamental de dicha señal.

Se puede pensar como sección de una señal periódica infinita. Sabemos que una señal con una única frecuencia es periódica y la suma de funciones periódicas son periódicas, con una frecuencia fundamental $f_0$.

Por consecuencia la señal se puede representar como :

$$
\mathcal{x(t)=x_1(t)+x_2(t)}
$$
Donde  $x_1$ tiene una frecuencia $f_1$, $x_2$ tiene una frecuencia $f_2$, los cuales son múltiplos de una frecuencia fundamental  $f_0$.

Dicho esto para poder obtener la frecuencia fundamenta de cada símbolo hay que hallar el máximo común divisor entre ambas.

```julia id=c10a69bb-8e59-493f-bb02-2db74abf6e2f
#Busco el máximo común divisor
SymbAf0=gcd(697,1633);
SymbBf0=gcd(770,1633);
SymbCf0=gcd(852,1633);
SymbDf0=gcd(941,1633);
Symb1f0=gcd(697,1209);
Symb2f0=gcd(697,1336);
Symb3f0=gcd(697,1477);
Symb4f0=gcd(770,1209);
Symb5f0=gcd(770,1336);
Symb6f0=gcd(770,1477);
Symb7f0=gcd(852,1209);
Symb8f0=gcd(852,1336);
Symb9f0=gcd(852,1477);
Symbastf0=gcd(941,1209);
Symb0f0=gcd(941,1209);
Symbhashf0=gcd(941,1209);
```

A continuación se procede a mostrar una tabla de símbolos y frecuencias fundamentales

![Tabla.jpg][nextjournal#file#d7218fa5-abe0-419f-884c-75df324c0184]

## 2. Caracterización espectral

En primer instancia procedo a graficar el módulo del espectro para la señal completa.

Claramente no se puede identificar los símbolos por separado ya que en la señal original hay ruido presente.

```julia id=2c220c20-317a-448d-a162-3c6742c9773f
fs=range(0; stop=sr, length=length(x))
xfft=fft(x)
xfftabs=abs.(xfft)
plot(fs, xfftabs,
  xlabel = "f (Hz)",
  title = "Módulo del espectro de la señal",xticks=0:200:sr/2,xlim=(0,4000))

```

![result][nextjournal#output#2c220c20-317a-448d-a162-3c6742c9773f#result]

Armo todos los vectores necesarios. Solo comento para el primer tono , el resto es muy similar.

```julia id=f3d6773e-3683-4f83-9472-a47bd0994ab9
#Armo un vector vacío
x1=zeros(0)
#Tomo los valores de la primer señal que corresponden al primer tono
x1=x[1719:2800]


#Tomo la fft de mi señal ya "recortada"

x1fft=fft(x1)

#=
fs1=range(0;length=length(x1fft),step=sr/length(x1fft))


let  
  
  fs1_aux= cshift.(fftshift(fs1), sr,from=-sr/2)
  p1=plot(fs1_aux, fftshift(abs.(x1fft)),xticks =600:75:1700,xlims=(600, 1700));
  
end
=#

fs1=range(0; stop=sr, length=length(x1));

x1fft=fft(x1);

maxfft1val=maximum(abs.(x1fft))
x1fft_Norm=fft(x1)/maxfft1val

x1fftabs=abs.(x1fft_Norm);

#Tono 2

x2=zeros(0)
x2=x[4640:5310]


fs2=range(0; stop=sr, length=length(x2));

x2fft=fft(x2);

maxfft2val=maximum(abs.(x2fft))
x2fft_Norm=fft(x2)/maxfft2val

x2fftabs=abs.(x2fft_Norm);

#Tono 3

x3=zeros(0)
x3=x[8080:9040]


fs3=range(0; stop=sr, length=length(x3));

x3fft=fft(x3);

maxfft3val=maximum(abs.(x3fft))
x3fft_Norm=fft(x3)/maxfft3val

x3fftabs=abs.(x3fft_Norm);

#Tono 4




x4=zeros(0)
x4=x[11600:12480]



fs4=range(0; stop=sr, length=length(x4));

x4fft=fft(x4);

maxfft4val=maximum(abs.(x4fft))
x4fft_Norm=fft(x4)/maxfft4val

x4fftabs=abs.(x4fft_Norm);



x5=zeros(0)
x5=x[14880:15840]



fs5=range(0; stop=sr, length=length(x5));

x5fft=fft(x5);

maxfft5val=maximum(abs.(x5fft))

x5fft_Norm=fft(x5)/maxfft5val

x5fftabs=abs.(x5fft_Norm);


x6=zeros(0)
x6=x[18480:19600]


fs6=range(0; stop=sr, length=length(x6));

x6fft=fft(x6);

maxfft6val=maximum(abs.(x6fft))

x6fft_Norm=fft(x6)/maxfft6val

x6fftabs=abs.(x6fft_Norm);
```

Dejo preparado todos los plots para poder comparar fácilmente

```julia id=558d8069-82ab-4023-80ff-94413a64f8ab
p1=plot(fs1, x1fftabs,xlabel = "f (Hz)",title = "Módulo del espectro del Primer tono",xticks=0:220:sr/2,xlims=(0, sr/2),linecolor=:red);
p2=plot(fs2, x2fftabs,xlabel = "f (Hz)",title = "Módulo del espectro del Segundo tono",xticks =0:220:sr/2,xlims=(0, sr/2));
p3=plot(fs3, x3fftabs,xlabel = "f (Hz)",title = "Módulo del espectro del Tercer tono",xticks=0:220:sr/2,xlims=(0, sr/2),linecolor=:red);
p4=plot(fs4, x4fftabs,xlabel = "f (Hz)",title = "Módulo del espectro del Cuarto tono",xticks=0:220:sr/2,xlims=(0, sr/2));
p5=plot(fs5, x5fftabs,xlabel = "f (Hz)",title = "Módulo del espectro del Quinto tono",xticks=0:220:sr/2,xlims=(0, sr/2),linecolor=:green);
p6=plot(fs6, x6fftabs,xlabel = "f (Hz)",title = "Módulo del espectro del Sexto tono",xticks=0:220:sr/2,xlims=(0, sr/2),linecolor=:black);
```

```julia id=0f7240a8-eb17-47a7-aab5-e4d1d8d44e7c
plot(p1, p2, layout = (2, 1), legend = false)
```

![result][nextjournal#output#0f7240a8-eb17-47a7-aab5-e4d1d8d44e7c#result]

Se puede observar a través de los gráficos anteriores que el primer símbolo y el segundo son el mismo.

```julia id=14b21fb6-2d54-4f4e-8591-a31999537f85
plot(p3, p5, layout = (2, 1), legend = false)
```

![result][nextjournal#output#14b21fb6-2d54-4f4e-8591-a31999537f85#result]

De la misma forma que con los gráficos anteriores podemos ver que el tercer símbolo y quinto son iguales.

```julia id=064bdb63-8d97-47eb-8ce1-23fa2fd8c824
display(p4)
```

![result][nextjournal#output#064bdb63-8d97-47eb-8ce1-23fa2fd8c824#result]

```julia id=50f9c181-d22e-4149-87a3-fa865be97dad
display(p6)
```

![result][nextjournal#output#50f9c181-d22e-4149-87a3-fa865be97dad#result]

Por último vemos que el cuarto y sexto símbolo son distintos entre si y  a todos los demás mostrados.

## 3. Espectrograma

Se procede a armar dos ventanas donde lo único que cambia es el ancho de la misma

```julia id=9ab3c8d0-c0a3-4c8b-b64d-ad4ce253f6c6
sprect=specplot(x;overlap=0.8,window=rect(300;padding=0,zerophase=false), fs=sr,yticks=400:50:1700,ylims=(400, 1700),title="Espectrograma rectangular angosta",xticks=0:0.2:3);
sprect2=specplot(x;overlap=0.8,window=rect(800;padding=0,zerophase=false), fs=sr,yticks=400:50:1700,ylims=(400, 1700),title="Espectrograma rectangular ancha",xticks=0:0.2:3);



```

```julia id=042e9e2f-16f6-4652-829a-d6f44a61862e
sprect
```

![result][nextjournal#output#042e9e2f-16f6-4652-829a-d6f44a61862e#result]

```julia id=d849cb78-0489-4974-a8c9-90e3780bad30
sprect2
```

![result][nextjournal#output#d849cb78-0489-4974-a8c9-90e3780bad30#result]

Ambos espectrogramas se realizaron con una ventana rectangular pero se tomaron diferentes cantidades de puntos. En el primer espectrograma se puede ver una buena resolución en tiempo viendo bien claro donde están los silencios, por ejemplo de 0.8 segundos a 1 segundo, pero una mala resolución en frecuencia. Para el segundo espectrograma tenemos el resultado contrario , una buena resolución en frecuencia pero mala en tiempos.

Ahora voy a proceder a cambiar el tipo de ventana manteniendo el ancho.

```julia id=e3dc35c2-bde3-41c8-b38b-b8d8114b6167
sphamm=specplot(x;overlap=0.8,window=hamming(800;padding=0,zerophase=false), fs=sr,yticks=400:50:1700,ylims=(400, 1700),title="Espectrograma hamming ",xticks=0:0.2:3)
```

![result][nextjournal#output#e3dc35c2-bde3-41c8-b38b-b8d8114b6167#result]

Se puede observar que en el espectrograma de la ventana rectangular("ancha") que el lóbulo principal es mas angosto, sin embargo los  lóbulos secundarios son mas altos por lo que complejiza la lectura del lóbulo principal,pero una vez identificado se puede ver una frecuencia marcada.Por otro lado para la ventana de hamming obtenemos el resultado opuesto,los lóbulos principales son mas anchos, por lo que no se puede ver una frecuencia precisa aunque casi no se pueden observar lóbulos secundarios muy altos.

## 4. Muestreo

Según el teorema de muestreo de Nyquist una señal, en este caso x(t), el muestreo debe cumplir con :

$$
w_s=\tfrac{2 \pi}{T} \geq 2W
$$
Hay que buscar que la frecuencia de muestreo sea al menos igual a la máxima frecuencia presente en DTMF, es decir mayor que  1633 Hz. A su vez también debe cumplir con Nyquist como ya fue mencionado previamente. En frecuencia implica, que para que se pueda muestrear correctamente la frecuencia mínima debe ser:

$$
f= 3266\quad Hz
$$
#### Sabiendo que las señales discretas se obtuvieron muestreando las originales con $F_S=8000Hz$, calcular los valores para las frecuencias DTMF presentes en el campo discreto.

Al no haber aliasing,la transformada de Fourier del muestreo $x[n]$, es una versión escalada en frecuencia de  $X_c(jw)$, ese valor de escala esta dado por :

$$
wT_s=\Omega
$$
Por lo que se procede a mostrar la siguiente tabla:

![Tabla2.png.jpg][nextjournal#file#71237272-f2ba-4781-a4c4-f0efce96cfe5]

## 5. Detección de dígitos

```julia id=9b97edd3-8ccf-4d87-82af-52cff4f513dc
sphamm
```

![result][nextjournal#output#9b97edd3-8ccf-4d87-82af-52cff4f513dc#result]

Con el espectrograma podemos observar que el primer tono posee 2 frecuencias : la primera entre 650 Hz y 750 Hz, y la segunda entre 1200 Hz y 1250 Hz. Para mas información acudimos a la transformada.

```julia id=ec4eedb4-5b29-4ab4-bd25-507ce4b635be
p1
```

![result][nextjournal#output#ec4eedb4-5b29-4ab4-bd25-507ce4b635be#result]

Finalmente podemos concluir que el primer tono corresponde a un 1

Al observar el siguiente gráfico a continuación se puede apreciar que es muy similar al anterior y en el espectrograma se ve un comportamiento similar, por lo que se llega a la conclusión que el segundo tono es nuevamente un 1

```julia id=9df6d2a9-6d38-474b-88ea-667f5f938d8c
p2
```

![result][nextjournal#output#9df6d2a9-6d38-474b-88ea-667f5f938d8c#result]

Para el tercer símbolo el espectrograma nos muestra una primer  frecuencia entre 900 Hz y 950 Hz y una segunda frecuencia entre 1300 y 1350 Hz. Si utilizamos también el gráfico a continuación se puede notar que el tercer tono corresponde a 0

```julia id=a49d697c-0d50-4142-a352-2a8d08419ab6
p3
```

![result][nextjournal#output#a49d697c-0d50-4142-a352-2a8d08419ab6#result]

Para el cuarto símbolo trabajando de la misma manera y se detecta que es un 4

```julia id=af392dc6-e445-4676-aff2-6af628469435
p4
```

![result][nextjournal#output#af392dc6-e445-4676-aff2-6af628469435#result]

En el punto 2 se mostró la comparativa entre el primer  y segundo tono, y también entre el tercero y quinto. Se pudo observar en su momento que la transformada era muy similar. Ahora si observamos el espectrograma vemos que el quinto símbolo tiene frecuencias muy similares entre ellas por lo que concluimos que el quinto símbolo es un 0, al igual que el tercero.

Finalmente para el sexto tono utilizamos el espectrograma y la transformada a continuación

```julia id=5cbcbe39-2999-4038-aa24-ede86e1f9cb7
p6
```

![result][nextjournal#output#5cbcbe39-2999-4038-aa24-ede86e1f9cb7#result]

Donde concluimos que el sexto tono corresponde a un 5.

La secuencia de tonos que se marco fue la siguiente : 

$$
 1-1-0-4-0-5.
$$

#### Mostrar la ventana de menor N con la que se puede todavía discriminar las 2 frecuencias del primer dígito. ¿Cuál es la ventana de mayor N que se puede utilizar? Mostrar los espectrogramas o las DFT para justificar.

Dado que cada tono es la suma de dos frecuencias específicas, si en el espectrograma una de esas frecuencias es incierta y se confunde con alguna otra de la tabla DTMF puede prestar a la confusión. Por ejemplo, para el tono "1" las frecuencias son $697\mathrm{Hz}$ y $1209\mathrm{Hz}$,si el ancho del lóbulo principal es mas grande y por ejemplo en la que correspondería a  $697\mathrm{Hz}$  llegase a tomar valores entre  $600\mathrm{Hz}$  y  $800\mathrm{Hz}$ puede llegar a interpretarse como un "4" en vez de un "1".

En el siguiente espectrograma, mostrado en el punto 3, se puede observar que para el primer tono la frecuencia mas baja esta entre  $650 \mathrm{Hz}$ y $700 \mathrm{Hz}$  , y la frecuencia mas alta entre  $1200 \mathrm{Hz}$ y $1250 \mathrm{Hz}$. Esta mas que claro que el tono corresponde a "1". Para este tono el problema lo va a presentar la frecuencia baja ya que, la diferencia para que se confunda con el próximo valor de interés,que es $770 \mathrm{Hz}$ , $\Delta_f= 73 Hz$.

```julia id=28632111-2518-41e2-a685-3484234160e3
sphamm=specplot(x;overlap=0.8,window=hamming(800;padding=0,zerophase=false), fs=sr,yticks=400:50:1700,ylims=(400, 1700),title="Espectrograma hamming ",xticks=0:0.2:3)
```

![result][nextjournal#output#28632111-2518-41e2-a685-3484234160e3#result]

Utilizo como criterio que el ancho máximo que se le va a agregar al lo por tomar un N mínimo es de $\tfrac{\Delta_f}{2}= 36,5 Hz$. Luego pasando este valor a tiempo y teniendo en cuenta que $f_s=8KHz$, obtenemos que :

$$
N_{min}=f_s\cdot\tfrac{2}{\Delta_f} \approx 219,18
$$
Por ende se va a aproximar a 220. En el siguiente gráfico se procede a mostrar el resultado analizado previamente

```julia id=b447f57a-1747-46c2-83e7-8115ac61afb2
sphamm=specplot(x1;overlap=0.8,window=hamming(220;padding=0,zerophase=false), fs=sr,yticks=400:50:1700,ylims=(400, 1700),title="Espectrograma con N mínimo ",xticks=0:0.2:3)
```

![result][nextjournal#output#b447f57a-1747-46c2-83e7-8115ac61afb2#result]

A fines prácticos podemos observar que el N podría ser mas chico. Para esto se itero hasta hallar un N mínimo acorde a lo pedido

```julia id=594974ba-13f3-4108-868a-faff9fbf63e3
sphamm=specplot(x1;overlap=0.8,window=hamming(120;padding=0,zerophase=false), fs=sr,yticks=400:50:1700,ylims=(400, 1700),title="Espectrograma con N mínimo iterado ",xticks=0:0.2:3)
```

![result][nextjournal#output#594974ba-13f3-4108-868a-faff9fbf63e3#result]

En el espectrograma se aprecia correctamente que el lóbulo principal de  la frecuencia baja del primer tono va desde $650 \mathrm{Hz}$ hasta cuanto mucho $750 \mathrm{Hz}$ por lo que cumple con lo pedido.

En el inciso 3 se hablo que si aumentamos el N mejoramos la resolución en frecuencia pero se empeora en tiempo. Un $N_{max}$ es aquel tal que permite identificar cuando empieza un tono y termina el otro. Para averiguar dicho numero tomamos la diferencia de tiempo entre el comienzo del segundo tono y la finalización del primero:$\Delta_t= 230 ms$. Por lo que el $N_{max}$ se procede a calcular como :

$$
N_{max}=\Delta_t \cdot f_s=1840
$$
El siguiente espectrograma se procede a graficarlo con dicho N

```julia id=802593d5-3031-41a1-b6da-f89db94f4f2e
sphamm=specplot(x;overlap=0.8,window=hamming(1840;padding=0,zerophase=false), fs=sr,yticks=400:50:1700,ylims=(400, 1700),title="Espectrograma con N  máximo",xticks=0:0.2:3)
```

![result][nextjournal#output#802593d5-3031-41a1-b6da-f89db94f4f2e#result]

#### Uno de los métodos que podría utilizarse para decodificar automáticamente los tonos consiste en un banco de filtros pasabanda centrados en cada una de las frecuencias del sistema DTMF. Diseñe el filtro que corresponde a una de las frecuencias del primer tono de la señal por el [método de ventaneo](https://nextjournal.com/sys-fiuba/dise%C3%B1o-de-filtros-digitales-fir). Justificar la elección del ancho de banda, la longitud, el tipo de ventana. 

#### Realizar el diagrama de polos y ceros. Y mostrar cuales son los coeficientes del mismo para filtrar con la función *`filt`*. 

Con el software **Pyfdax** armo un filtro con frecuencia de corte 660Hz y 720Hz para  que entre únicamente la frecuencia 697 Hz

![Filtro.jpg][nextjournal#file#6fdbe178-6447-492d-89a1-c032c78d864a]

Podemos observar que los lóbulos secundarios son cada vez menores.

[filtroBp.npz][nextjournal#file#5aedebcc-628a-4134-9a5c-c5ef0de84372]

```python id=140f871a-064f-45ff-8d29-a4e1ad0f10fe
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
```

```python id=bf44e8c3-bd93-4926-8b46-688516d55d83
#Abro el archivo
fspy, xpy=wavfile.read( [reference][nextjournal#reference#77f49149-5108-414f-9311-072688707cf8]);
with np.load([reference][nextjournal#reference#dbf7082d-2049-4494-a876-8ad6e01ad6b0]) as filter_bandpass:
  ba=filter_bandpass['ba']
print(np.matrix(ba[0,:]))
```

Los coeficientes son 

\[\[-1.26398924e-05 4.21198706e-05 8.39967297e-05 1.01258237e-04 8.94104138e-05 5.23634888e-05 1.16128192e-06 -4.93184010e-05 -8.47051187e-05 -9.52473146e-05 -7.85117268e-05 -3.99193325e-05 8.99214685e-06 5.40278294e-05 8.24779767e-05 8.67281822e-05 6.63049736e-05 2.78040401e-05 -1.72214100e-05 -5.57419280e-05 -7.70507075e-05 -7.57647005e-05 -5.31787975e-05 -1.66328176e-05 2.28542250e-05 5.39263945e-05 6.81914562e-05 6.25107256e-05 3.96453172e-05 7.13982295e-06 -2.51444851e-05 -4.80476938e-05 -5.57508147e-05 -4.72643736e-05 -2.63801777e-05 -1.89198839e-07 2.32957234e-05 3.76248075e-05 3.97255246e-05 3.05223698e-05 1.42483566e-05 -3.23803966e-06 -1.65107241e-05 -2.23003386e-05 -2.03280035e-05 -1.30243251e-05 -4.30692918e-06 2.08574781e-06 4.06588222e-06 1.92404468e-06 -1.94856646e-06 -4.22331138e-06 -2.22204396e-06 4.70588845e-06 1.45948249e-05 2.33609815e-05 2.62808664e-05 1.99526969e-05 4.01232929e-06 -1.81019777e-05 -3.97548911e-05 -5.30412265e-05 -5.14928192e-05 -3.26761050e-05 2.71959275e-07 3.88544265e-05 7.13208627e-05 8.61772856e-05 7.60691581e-05 4.07666876e-05 -1.18523787e-05 -6.73737687e-05 -1.08731796e-04 -1.21381940e-04 -9.82117511e-05 -4.25774187e-05 3.16912384e-05 1.03604666e-04 1.50896491e-04 1.56841693e-04 1.15941276e-04 3.65924875e-05 -6.03381082e-05 -1.46920179e-04 -1.96171459e-04 -1.90388304e-04 -1.27242096e-04 -2.16008099e-05 9.77830743e-05 1.96050228e-04 2.42390127e-04 2.19622385e-04 1.30243081e-04 -3.12273197e-06 -1.43333743e-04 -2.49058390e-04 -2.86949897e-04 -2.42085819e-04 -1.23422172e-04 3.76457610e-05 1.95533107e-04 3.03378247e-04 3.26958626e-04 2.55474225e-04 1.05818140e-04 -8.12356599e-05 -2.52133758e-04 -3.55916100e-04 -3.59436275e-04 -2.57875289e-04 -7.72298361e-05 1.32264593e-04 3.10140547e-04 4.03221212e-04 3.81561349e-04 2.48014153e-04 3.83816210e-05 -1.88179312e-04 -3.65928825e-04 -4.41718323e-04 -3.90945935e-04 -2.25483744e-04 8.96610731e-06 2.45548912e-04 4.15439140e-04 4.67990490e-04 3.85918156e-04 1.90936254e-04 -6.19800242e-05 -3.00198147e-04 -4.54442174e-04 -4.79093950e-04 -3.65787303e-04 -1.46212433e-04 1.16805073e-04 3.47427010e-04 4.78860407e-04 4.72881338e-04 3.31065221e-04 9.43879522e-05 -1.68712018e-04 -3.82309507e-04 -4.85126069e-04 -4.48305742e-04 -2.83618072e-04 -3.97209086e-05 2.12342047e-04 4.00056550e-04 4.70549081e-04 4.05676317e-04 2.26725491e-04 -1.25087032e-05 -2.42040600e-04 -3.96420367e-04 -4.33664549e-04 -3.46836754e-04 -1.65029450e-04 5.62684057e-05 2.52263850e-04 3.68111401e-04 3.74527403e-04 2.75241142e-04 1.04362505e-04 -8.51431122e-05 -2.38033045e-04 -3.13194157e-04 -2.94922440e-04 -1.95907443e-04 -5.14541798e-05 9.27874820e-05 1.95404882e-04 2.31426247e-04 1.98461456e-04 1.15236838e-04 1.35243377e-05 -7.34337630e-05 -1.21921080e-04 -1.24505548e-04 -9.05452990e-05 -4.06970209e-05 2.21718738e-06 2.24204437e-05 1.69978647e-05 -3.80602770e-06 -2.18226560e-05 -1.96214946e-05 1.08644251e-05 6.32985417e-05 1.17783416e-04 1.47707053e-04 1.30374522e-04 5.75556727e-05 -5.82369642e-05 -1.84707160e-04 -2.78519541e-04 -2.99545053e-04 -2.26058415e-04 -6.54956784e-05 1.43625945e-04 3.40385906e-04 4.58990604e-04 4.50173170e-04 2.99640173e-04 3.70455467e-05 -2.68503265e-04 -5.26314619e-04 -6.50845343e-04 -5.89466440e-04 -3.42393203e-04 3.23286573e-05 4.31686757e-04 7.35851127e-04 8.43972626e-04 7.06975751e-04 3.46833126e-04 -1.44714526e-04 -6.29093838e-04 -9.59906954e-04 -1.02705131e-03 -7.92684720e-04 -3.07446051e-04 2.99284423e-04 8.53684931e-04 1.18732833e-03 1.18825624e-03 8.37823396e-04 2.21354781e-04 -4.91989882e-04 -1.09562139e-03 -1.40547595e-03 -1.31608424e-03 -8.35683937e-04 -8.88664979e-05 7.15459484e-04 1.34264894e-03 1.60098148e-03 1.40025134e-03 7.82378038e-04 -8.61509730e-05 -9.59128993e-04 -1.58070190e-03 -1.76064334e-03 -1.43260287e-03 -6.77474592e-04 2.96109503e-04 1.20961809e-03 1.79470659e-03 1.87241105e-03 1.40797152e-03 5.24459236e-04 -5.29791002e-04 -1.45135130e-03 -1.96954625e-03 -1.92639591e-03 -1.32491669e-03 -3.30965155e-04 7.72706042e-04 1.66740284e-03 2.09113421e-03 1.91583887e-03 1.18628086e-03 1.08736542e-04 -1.00772135e-03 -1.84052724e-03 -2.14753028e-03 -1.83796319e-03 -9.99506592e-04 1.26698136e-04 1.21593775e-03 1.95432152e-03 2.13002657e-03 1.69464193e-03 7.76669691e-04 -3.56647401e-04 -1.37778094e-03 -1.99445033e-03 -2.03412489e-03 -1.49281722e-03 -5.34200487e-04 5.60156638e-04 1.47424971e-03 1.94985648e-03 1.86032951e-03 1.24462077e-03 2.92284380e-04 -7.15150203e-04 -1.48825084e-03 -1.81387339e-03 -1.61468565e-03 -9.67161423e-04 -7.39542749e-05 7.99759968e-04 1.40593884e-03 1.58515689e-03 1.30900606e-03 6.81965836e-04 -9.60906916e-05 -7.93767737e-04 -1.21797208e-03 -1.26835978e-03 -9.60745169e-04 -4.14080660e-04 1.92884773e-04 6.80076068e-04 9.20596167e-04 8.74484238e-04 5.92500726e-04 1.90867812e-04 -1.92739766e-04 -4.46114153e-04 -5.16470944e-04 -4.20864173e-04 -2.31146272e-04 -4.05469328e-05 7.48453082e-05 8.50834295e-05 1.51683502e-05 -6.92491454e-05 -9.33776699e-05 -9.44073148e-06 1.77172551e-04 4.03048189e-04 5.66714375e-04 5.67500868e-04 3.49561835e-04 -6.58436027e-05 -5.73835857e-04 -1.01088392e-03 -1.20586050e-03 -1.04162256e-03 -5.06042101e-04 2.87126057e-04 1.11864063e-03 1.72333154e-03 1.87296794e-03 1.45694980e-03 5.33437015e-04 -6.69330707e-04 -1.80721269e-03 -2.51777509e-03 -2.53389435e-03 -1.77819499e-03 -4.06203904e-04 1.22021631e-03 2.62686613e-03 3.36470180e-03 3.15116873e-03 1.97138305e-03 1.04401591e-04 -1.93933584e-03 -3.55661541e-03 -4.22877351e-03 -3.68579987e-03 -2.00584255e-03 3.84752191e-04 2.81740034e-03 4.56766673e-03 5.07030357e-03 4.09928968e-03 1.85613587e-03 -1.06563333e-03 -3.83610736e-03 -5.62438522e-03 -5.84707311e-03 -4.35574289e-03 -1.50381063e-03 1.93336424e-03 4.96846850e-03 6.68570566e-03 6.51639679e-03 4.42395309e-03 9.38859963e-04 -2.97343694e-03 -6.17964142e-03 -7.70692532e-03 -7.03732881e-03 -4.27934208e-03 -1.60792505e-04 4.16185032e-03 7.42824102e-03 8.64179200e-03 7.37288731e-03 3.90563301e-03 -8.20767384e-04 -5.46577634e-03 -8.66807484e-03 -9.44477920e-03 -7.49216916e-03 -3.29614976e-03 1.98600316e-03 6.84473817e-03 9.85021973e-03 1.00734251e-02 7.37222975e-03 2.45465339e-03 -3.30537793e-03 -8.25225249e-03 -1.09253342e-02 -1.04906040e-02 -6.99961178e-03 -1.39565153e-03 4.74064696e-03 9.63785859e-03 1.18460826e-02 1.06666000e-02 6.37142496e-03 1.44145924e-04 -6.24637126e-03 -1.09494320e-02 -1.25695392e-02 -1.05808579e-02 -5.49590202e-03 1.26518438e-03 7.77186148e-03 1.21356609e-02 1.30594341e-02 1.02233068e-02 4.39238580e-03 -2.78933359e-03 -9.26345485e-03 -1.31485503e-02 -1.32881147e-02 -9.59516978e-03 -3.09073427e-03 4.37879059e-03 1.06670059e-02 1.39458171e-02 1.32381045e-02 8.70920481e-03 1.63016440e-03 -5.97974016e-03 -1.19304574e-02 -1.44930386e-02 -1.29031672e-02 -7.58935250e-03 -5.75885260e-05 7.53652868e-03 1.30063503e-02 1.47654341e-02 1.22888090e-02 6.26980121e-03 -1.57447252e-03 -8.99426261e-03 -1.38541345e-02 -1.47491757e-02 -1.14121833e-02 -4.79351359e-03 3.21029009e-03 1.03013986e-02 1.44421512e-02 1.44421512e-02 1.03013986e-02 3.21029009e-03 -4.79351359e-03 -1.14121833e-02 -1.47491757e-02 -1.38541345e-02 -8.99426261e-03 -1.57447252e-03 6.26980121e-03 1.22888090e-02 1.47654341e-02 1.30063503e-02 7.53652868e-03 -5.75885260e-05 -7.58935250e-03 -1.29031672e-02 -1.44930386e-02 -1.19304574e-02 -5.97974016e-03 1.63016440e-03 8.70920481e-03 1.32381045e-02 1.39458171e-02 1.06670059e-02 4.37879059e-03 -3.09073427e-03 -9.59516978e-03 -1.32881147e-02 -1.31485503e-02 -9.26345485e-03 -2.78933359e-03 4.39238580e-03 1.02233068e-02 1.30594341e-02 1.21356609e-02 7.77186148e-03 1.26518438e-03 -5.49590202e-03 -1.05808579e-02 -1.25695392e-02 -1.09494320e-02 -6.24637126e-03 1.44145924e-04 6.37142496e-03 1.06666000e-02 1.18460826e-02 9.63785859e-03 4.74064696e-03 -1.39565153e-03 -6.99961178e-03 -1.04906040e-02 -1.09253342e-02 -8.25225249e-03 -3.30537793e-03 2.45465339e-03 7.37222975e-03 1.00734251e-02 9.85021973e-03 6.84473817e-03 1.98600316e-03 -3.29614976e-03 -7.49216916e-03 -9.44477920e-03 -8.66807484e-03 -5.46577634e-03 -8.20767384e-04 3.90563301e-03 7.37288731e-03 8.64179200e-03 7.42824102e-03 4.16185032e-03 -1.60792505e-04 -4.27934208e-03 -7.03732881e-03 -7.70692532e-03 -6.17964142e-03 -2.97343694e-03 9.38859963e-04 4.42395309e-03 6.51639679e-03 6.68570566e-03 4.96846850e-03 1.93336424e-03 -1.50381063e-03 -4.35574289e-03 -5.84707311e-03 -5.62438522e-03 -3.83610736e-03 -1.06563333e-03 1.85613587e-03 4.09928968e-03 5.07030357e-03 4.56766673e-03 2.81740034e-03 3.84752191e-04 -2.00584255e-03 -3.68579987e-03 -4.22877351e-03 -3.55661541e-03 -1.93933584e-03 1.04401591e-04 1.97138305e-03 3.15116873e-03 3.36470180e-03 2.62686613e-03 1.22021631e-03 -4.06203904e-04 -1.77819499e-03 -2.53389435e-03 -2.51777509e-03 -1.80721269e-03 -6.69330707e-04 5.33437015e-04 1.45694980e-03 1.87296794e-03 1.72333154e-03 1.11864063e-03 2.87126057e-04 -5.06042101e-04 -1.04162256e-03 -1.20586050e-03 -1.01088392e-03 -5.73835857e-04 -6.58436027e-05 3.49561835e-04 5.67500868e-04 5.66714375e-04 4.03048189e-04 1.77172551e-04 -9.44073148e-06 -9.33776699e-05 -6.92491454e-05 1.51683502e-05 8.50834295e-05 7.48453082e-05 -4.05469328e-05 -2.31146272e-04 -4.20864173e-04 -5.16470944e-04 -4.46114153e-04 -1.92739766e-04 1.90867812e-04 5.92500726e-04 8.74484238e-04 9.20596167e-04 6.80076068e-04 1.92884773e-04 -4.14080660e-04 -9.60745169e-04 -1.26835978e-03 -1.21797208e-03 -7.93767737e-04 -9.60906916e-05 6.81965836e-04 1.30900606e-03 1.58515689e-03 1.40593884e-03 7.99759968e-04 -7.39542749e-05 -9.67161423e-04 -1.61468565e-03 -1.81387339e-03 -1.48825084e-03 -7.15150203e-04 2.92284380e-04 1.24462077e-03 1.86032951e-03 1.94985648e-03 1.47424971e-03 5.60156638e-04 -5.34200487e-04 -1.49281722e-03 -2.03412489e-03 -1.99445033e-03 -1.37778094e-03 -3.56647401e-04 7.76669691e-04 1.69464193e-03 2.13002657e-03 1.95432152e-03 1.21593775e-03 1.26698136e-04 -9.99506592e-04 -1.83796319e-03 -2.14753028e-03 -1.84052724e-03 -1.00772135e-03 1.08736542e-04 1.18628086e-03 1.91583887e-03 2.09113421e-03 1.66740284e-03 7.72706042e-04 -3.30965155e-04 -1.32491669e-03 -1.92639591e-03 -1.96954625e-03 -1.45135130e-03 -5.29791002e-04 5.24459236e-04 1.40797152e-03 1.87241105e-03 1.79470659e-03 1.20961809e-03 2.96109503e-04 -6.77474592e-04 -1.43260287e-03 -1.76064334e-03 -1.58070190e-03 -9.59128993e-04 -8.61509730e-05 7.82378038e-04 1.40025134e-03 1.60098148e-03 1.34264894e-03 7.15459484e-04 -8.88664979e-05 -8.35683937e-04 -1.31608424e-03 -1.40547595e-03 -1.09562139e-03 -4.91989882e-04 2.21354781e-04 8.37823396e-04 1.18825624e-03 1.18732833e-03 8.53684931e-04 2.99284423e-04 -3.07446051e-04 -7.92684720e-04 -1.02705131e-03 -9.59906954e-04 -6.29093838e-04 -1.44714526e-04 3.46833126e-04 7.06975751e-04 8.43972626e-04 7.35851127e-04 4.31686757e-04 3.23286573e-05 -3.42393203e-04 -5.89466440e-04 -6.50845343e-04 -5.26314619e-04 -2.68503265e-04 3.70455467e-05 2.99640173e-04 4.50173170e-04 4.58990604e-04 3.40385906e-04 1.43625945e-04 -6.54956784e-05 -2.26058415e-04 -2.99545053e-04 -2.78519541e-04 -1.84707160e-04 -5.82369642e-05 5.75556727e-05 1.30374522e-04 1.47707053e-04 1.17783416e-04 6.32985417e-05 1.08644251e-05 -1.96214946e-05 -2.18226560e-05 -3.80602770e-06 1.69978647e-05 2.24204437e-05 2.21718738e-06 -4.06970209e-05 -9.05452990e-05 -1.24505548e-04 -1.21921080e-04 -7.34337630e-05 1.35243377e-05 1.15236838e-04 1.98461456e-04 2.31426247e-04 1.95404882e-04 9.27874820e-05 -5.14541798e-05 -1.95907443e-04 -2.94922440e-04 -3.13194157e-04 -2.38033045e-04 -8.51431122e-05 1.04362505e-04 2.75241142e-04 3.74527403e-04 3.68111401e-04 2.52263850e-04 5.62684057e-05 -1.65029450e-04 -3.46836754e-04 -4.33664549e-04 -3.96420367e-04 -2.42040600e-04 -1.25087032e-05 2.26725491e-04 4.05676317e-04 4.70549081e-04 4.00056550e-04 2.12342047e-04 -3.97209086e-05 -2.83618072e-04 -4.48305742e-04 -4.85126069e-04 -3.82309507e-04 -1.68712018e-04 9.43879522e-05 3.31065221e-04 4.72881338e-04 4.78860407e-04 3.47427010e-04 1.16805073e-04 -1.46212433e-04 -3.65787303e-04 -4.79093950e-04 -4.54442174e-04 -3.00198147e-04 -6.19800242e-05 1.90936254e-04 3.85918156e-04 4.67990490e-04 4.15439140e-04 2.45548912e-04 8.96610731e-06 -2.25483744e-04 -3.90945935e-04 -4.41718323e-04 -3.65928825e-04 -1.88179312e-04 3.83816210e-05 2.48014153e-04 3.81561349e-04 4.03221212e-04 3.10140547e-04 1.32264593e-04 -7.72298361e-05 -2.57875289e-04 -3.59436275e-04 -3.55916100e-04 -2.52133758e-04 -8.12356599e-05 1.05818140e-04 2.55474225e-04 3.26958626e-04 3.03378247e-04 1.95533107e-04 3.76457610e-05 -1.23422172e-04 -2.42085819e-04 -2.86949897e-04 -2.49058390e-04 -1.43333743e-04 -3.12273197e-06 1.30243081e-04 2.19622385e-04 2.42390127e-04 1.96050228e-04 9.77830743e-05 -2.16008099e-05 -1.27242096e-04 -1.90388304e-04 -1.96171459e-04 -1.46920179e-04 -6.03381082e-05 3.65924875e-05 1.15941276e-04 1.56841693e-04 1.50896491e-04 1.03604666e-04 3.16912384e-05 -4.25774187e-05 -9.82117511e-05 -1.21381940e-04 -1.08731796e-04 -6.73737687e-05 -1.18523787e-05 4.07666876e-05 7.60691581e-05 8.61772856e-05 7.13208627e-05 3.88544265e-05 2.71959275e-07 -3.26761050e-05 -5.14928192e-05 -5.30412265e-05 -3.97548911e-05 -1.81019777e-05 4.01232929e-06 1.99526969e-05 2.62808664e-05 2.33609815e-05 1.45948249e-05 4.70588845e-06 -2.22204396e-06 -4.22331138e-06 -1.94856646e-06 1.92404468e-06 4.06588222e-06 2.08574781e-06 -4.30692918e-06 -1.30243251e-05 -2.03280035e-05 -2.23003386e-05 -1.65107241e-05 -3.23803966e-06 1.42483566e-05 3.05223698e-05 3.97255246e-05 3.76248075e-05 2.32957234e-05 -1.89198839e-07 -2.63801777e-05 -4.72643736e-05 -5.57508147e-05 -4.80476938e-05 -2.51444851e-05 7.13982295e-06 3.96453172e-05 6.25107256e-05 6.81914562e-05 5.39263945e-05 2.28542250e-05 -1.66328176e-05 -5.31787975e-05 -7.57647005e-05 -7.70507075e-05 -5.57419280e-05 -1.72214100e-05 2.78040401e-05 6.63049736e-05 8.67281822e-05 8.24779767e-05 5.40278294e-05 8.99214685e-06 -3.99193325e-05 -7.85117268e-05 -9.52473146e-05 -8.47051187e-05 -4.93184010e-05 1.16128192e-06 5.23634888e-05 8.94104138e-05 1.01258237e-04 8.39967297e-05 4.21198706e-05 -1.26398924e-05\]\]

```python id=d4e58430-754d-495c-90bf-f16d0ca72806


tpy = np.linspace(0,((len(xpy)-1)/fspy),len(xpy)) 
  
filter_697_in_signal= signal.lfilter(ba[0,0:],ba[1,0:],xpy)

fft_filter_signal = np.fft.fft(filter_697_in_signal)

fft_filter_signal_mod = np.abs(fft_filter_signal)

Omega_filter = np.linspace(0,fspy,len(filter_697_in_signal))


fft_filt_sig_plot = plt.figure(1)
ax=plt.gca();
ax.xaxis.set_major_locator(plt.MaxNLocator(15))
plt.plot(Omega_filter,fft_filter_signal_mod)
plt.xlim(0,4000)
plt.title('Módulo del espectro de la señal filtrada en 697 Hz')
plt.xlabel('Frecuencia [Hz]')
plt.grid()
plt.show()

fft_filt_sig_plot
```

![result][nextjournal#output#d4e58430-754d-495c-90bf-f16d0ca72806#result]

Podemos notar que el filtro efectivamente filtra en el rango de 660 Hz a 720 Hz. El gráfico no es precisamente una delta ya que existen frecuencias cercanas a 697 Hz en el discado generando que  se sumen.

```python id=6e6e422b-d669-4bfc-9ba5-868ef747dc79
filt697_sig_plot = plt.figure(2)
ax=plt.gca();
ax.xaxis.set_major_locator(plt.MaxNLocator(15))
plt.plot(tpy,filter_697_in_signal)
plt.title('Señal filtrada para 660-720 Hz')
plt.xlabel('Tiempo [s]')
plt.grid()
plt.show()

filt697_sig_plot

```

![result][nextjournal#output#6e6e422b-d669-4bfc-9ba5-868ef747dc79#result]

```julia id=c4b50706-6c14-4b84-b89d-9e753aee1a79
plot(time_sample,x,title="Señal sin filtrar",xticks=0:0.25:3)
```

![result][nextjournal#output#c4b50706-6c14-4b84-b89d-9e753aee1a79#result]

Se puede observar claramente que la frecuencia 697Hz esta presente en el primer y segundo tono. Dado que el filtro tiene un ancho de banda de aproximadamente de 60 Hz entran otras frecuencias distintas a 697 Hz. Este ancho de banda permite que no tome la frecuencia de 770Hz. Además con este ancho de banda se podría armar un filtro para todas las frecuencias, ya que la menor diferencia entre frecuencias del DTMF es de 73 Hz.

Dado que el orden de filtro es alto, alrededor de 900, se puede observar un delay en la señal. En la señal sin filtrar el primer tono se da antes de los 250 ms pero en la señal filtrada luego de esto

![Polos_ceros.jpeg][nextjournal#file#dfe3a4a9-acda-497b-a190-2bb4a20b9f85]

#### ¿Tiene fase lineal?

Es lineal en la frecuencia en donde se encuentra el pasa banda en alta.Si a la pendiente que se puede observar en el gráfico se le cambia el signo, nos indica el retardo que va a tener la señal, estos retardos van a ser todos iguales para las frecuencias que deja pasar el pasa banda. La señal luego de pasar por este pasa banda en las frecuencias de 660Hz -720Hz no se van a deformar.

![Fase.jpeg][nextjournal#file#77ee5a75-1b76-4398-babe-3ed96705d60a]


[nextjournal#file#426d1501-3a13-436b-8626-baaa90743947]:
<https://nextjournal.com/data/QmXUsJCQXWZdm3QScXCSxPsPEgpHB2LxFd6BuY4EUzSizS?content-type=image/png&node-id=426d1501-3a13-436b-8626-baaa90743947&filename=screenshot.png&node-kind=file>

[nextjournal#file#fa3fd099-8ada-43ea-a176-770134676358]:
<https://nextjournal.com/data/QmbEwS8qRdgk6mwPDjdwssof4P8i4BkmV5iNbXvbCLS9uu?content-type=audio/wav&node-id=fa3fd099-8ada-43ea-a176-770134676358&filename=dtmf2.wav&node-kind=file>

[nextjournal#reference#bb4c58bd-1bf1-406a-a857-e0bd58b42c39]:
<#nextjournal#reference#bb4c58bd-1bf1-406a-a857-e0bd58b42c39>

[nextjournal#output#2e2671f2-1b8d-4f1d-aecc-545d1271c929#out.wav]:
<https://nextjournal.com/data/QmcagavvY2c9cBtnifZRnQzSVvFJNfnu8VLrTjQSsdNinQ?content-type=audio/vnd.wave&node-id=2e2671f2-1b8d-4f1d-aecc-545d1271c929&filename=out.wav&node-kind=output>

[nextjournal#output#00f9cce1-407f-41f0-a89a-5df7b192c0c5#result]:
<https://nextjournal.com/data/Qmc3optwh9BPHQBhobMv4HuHnzoKyJPp7TF888esWYbFBC?content-type=image/svg%2Bxml&node-id=00f9cce1-407f-41f0-a89a-5df7b192c0c5&node-kind=output>

[nextjournal#output#6110fad1-f177-4d7e-ae38-da56153e339e#result]:
<https://nextjournal.com/data/QmbbzWmysWQdrB4kbo9xmNMY6Nv5vh4Twr2t2UEFbUXHM2?content-type=image/svg%2Bxml&node-id=6110fad1-f177-4d7e-ae38-da56153e339e&node-kind=output>

[nextjournal#output#827fc4fe-e33d-45c0-acb4-c0bbdee00cfd#result]:
<https://nextjournal.com/data/QmQkSAtDmfiXrS1ZAxkMhW5E1VFs62G2gojBK939f2QUQL?content-type=image/svg%2Bxml&node-id=827fc4fe-e33d-45c0-acb4-c0bbdee00cfd&node-kind=output>

[nextjournal#output#25028c9c-3a6b-4623-875a-bc484beaaa57#result]:
<https://nextjournal.com/data/QmXnTz63mkCFar7odVUVgp5zaUGaLD5VmsNPdxAEZEfMor?content-type=image/svg%2Bxml&node-id=25028c9c-3a6b-4623-875a-bc484beaaa57&node-kind=output>

[nextjournal#output#b8f245ea-096c-4994-9e53-2f90ace5dcdb#result]:
<https://nextjournal.com/data/QmXWGu3txXQP4VwP8eKUoi56CqNdefDe4Pwdi8CFwA6aKJ?content-type=image/svg%2Bxml&node-id=b8f245ea-096c-4994-9e53-2f90ace5dcdb&node-kind=output>

[nextjournal#output#e1c3e588-a573-42a7-b123-606ad7686ece#result]:
<https://nextjournal.com/data/QmYPQFW937yivNhWd2VzuV7D2AabhXmzLpCE7EgsJ5zJqD?content-type=image/svg%2Bxml&node-id=e1c3e588-a573-42a7-b123-606ad7686ece&node-kind=output>

[nextjournal#output#77e4bc62-63ab-413d-ba11-1c8f86326967#result]:
<https://nextjournal.com/data/QmaAnS7TaZ2EpFEqLMoxHxX9TpE54NNUqbRJG6r6G72SK3?content-type=image/svg%2Bxml&node-id=77e4bc62-63ab-413d-ba11-1c8f86326967&node-kind=output>

[nextjournal#output#49d678c4-bc8e-425b-829f-263bc17bf9da#result]:
<https://nextjournal.com/data/QmcFUYTk3Em8t3BEQWEf84nmbjW8PREcX99wV8M3nZGxt2?content-type=image/svg%2Bxml&node-id=49d678c4-bc8e-425b-829f-263bc17bf9da&node-kind=output>

[nextjournal#file#90d69ca0-3e5e-4430-8c1c-e56f9878b591]:
<https://nextjournal.com/data/QmTyzsXXMiUzqBdgSHeN9r3KGTUSCq9SZyg7RfENcXmgsu?content-type=image/jpeg&node-id=90d69ca0-3e5e-4430-8c1c-e56f9878b591&filename=ppiofin.jpg&node-kind=file>

[nextjournal#file#1d662761-2f85-493c-bb8b-b5b834d22e49]:
<https://nextjournal.com/data/QmSvaffJZt2kKKZ1967TEMWqw52qpzYraxNFq911Jeajgp?content-type=image/jpeg&node-id=1d662761-2f85-493c-bb8b-b5b834d22e49&filename=duracion.jpg&node-kind=file>

[nextjournal#file#d7218fa5-abe0-419f-884c-75df324c0184]:
<https://nextjournal.com/data/QmXNLUeaNGFyARw4hdvLgz1GZWHpGK5JeFpNfKN2cz6L3A?content-type=image/jpeg&node-id=d7218fa5-abe0-419f-884c-75df324c0184&filename=Tabla.jpg&node-kind=file>

[nextjournal#output#2c220c20-317a-448d-a162-3c6742c9773f#result]:
<https://nextjournal.com/data/QmRc7p3a9j6zfYhUuA3Kp2T8ZLySaaaUuPMuTh5qPUaptB?content-type=image/svg%2Bxml&node-id=2c220c20-317a-448d-a162-3c6742c9773f&node-kind=output>

[nextjournal#output#0f7240a8-eb17-47a7-aab5-e4d1d8d44e7c#result]:
<https://nextjournal.com/data/QmVp2FuzpxLHHTMjJF6cZmhbo8kmHPNYU1qvov11q9EzAP?content-type=image/svg%2Bxml&node-id=0f7240a8-eb17-47a7-aab5-e4d1d8d44e7c&node-kind=output>

[nextjournal#output#14b21fb6-2d54-4f4e-8591-a31999537f85#result]:
<https://nextjournal.com/data/QmR91vKsgpZqfKJuXjEeAWvXB6Ax8ckub5uVVQ5JLcDLXp?content-type=image/svg%2Bxml&node-id=14b21fb6-2d54-4f4e-8591-a31999537f85&node-kind=output>

[nextjournal#output#064bdb63-8d97-47eb-8ce1-23fa2fd8c824#result]:
<https://nextjournal.com/data/QmXAUfJhWcSXgSyj21NhqQRLgQCbWHK13JgkZXxjW1U5vH?content-type=image/svg%2Bxml&node-id=064bdb63-8d97-47eb-8ce1-23fa2fd8c824&node-kind=output>

[nextjournal#output#50f9c181-d22e-4149-87a3-fa865be97dad#result]:
<https://nextjournal.com/data/QmcMdXFsms7JfRkwagxExLE7kwpsDcKEn6nmqx2WkdGzoP?content-type=image/svg%2Bxml&node-id=50f9c181-d22e-4149-87a3-fa865be97dad&node-kind=output>

[nextjournal#output#042e9e2f-16f6-4652-829a-d6f44a61862e#result]:
<https://nextjournal.com/data/QmdA5fzrzRhgssKh5ht3Ku2wb5Br4oVtTstzcbRd2YxwLb?content-type=image/svg%2Bxml&node-id=042e9e2f-16f6-4652-829a-d6f44a61862e&node-kind=output>

[nextjournal#output#d849cb78-0489-4974-a8c9-90e3780bad30#result]:
<https://nextjournal.com/data/QmShVu5swupEVp5ysKrRnhDiiVrRWhPSqnUxEZ2AELjxtg?content-type=image/svg%2Bxml&node-id=d849cb78-0489-4974-a8c9-90e3780bad30&node-kind=output>

[nextjournal#output#e3dc35c2-bde3-41c8-b38b-b8d8114b6167#result]:
<https://nextjournal.com/data/QmPVGAB3ZwMMEuc2McMbm2PtcDYatCUrGzMWVNKjEmns91?content-type=image/svg%2Bxml&node-id=e3dc35c2-bde3-41c8-b38b-b8d8114b6167&node-kind=output>

[nextjournal#file#71237272-f2ba-4781-a4c4-f0efce96cfe5]:
<https://nextjournal.com/data/Qmcgudka442rpaUF9HgQ45XE1JJX5YLaxhuuTH45xsMfM4?content-type=image/jpeg&node-id=71237272-f2ba-4781-a4c4-f0efce96cfe5&filename=Tabla2.png.jpg&node-kind=file>

[nextjournal#output#9b97edd3-8ccf-4d87-82af-52cff4f513dc#result]:
<https://nextjournal.com/data/QmRFmq8PUrGodFRpnrH7cECppaFyzxMPBhxFS1XDbHvYab?content-type=image/svg%2Bxml&node-id=9b97edd3-8ccf-4d87-82af-52cff4f513dc&node-kind=output>

[nextjournal#output#ec4eedb4-5b29-4ab4-bd25-507ce4b635be#result]:
<https://nextjournal.com/data/Qma9Nrvx96bQwk2PNniTnAqXaTCkNV1n3iQUZuDeXATP9b?content-type=image/svg%2Bxml&node-id=ec4eedb4-5b29-4ab4-bd25-507ce4b635be&node-kind=output>

[nextjournal#output#9df6d2a9-6d38-474b-88ea-667f5f938d8c#result]:
<https://nextjournal.com/data/QmfV7wWU2gUDCqxTh73wRmqYyqA2VPw2EwV1VmmqtzvfG3?content-type=image/svg%2Bxml&node-id=9df6d2a9-6d38-474b-88ea-667f5f938d8c&node-kind=output>

[nextjournal#output#a49d697c-0d50-4142-a352-2a8d08419ab6#result]:
<https://nextjournal.com/data/QmSVme6YstLVnQ2JsLEf1w8fhNqaikx2DqVdP9WKgP6ZPq?content-type=image/svg%2Bxml&node-id=a49d697c-0d50-4142-a352-2a8d08419ab6&node-kind=output>

[nextjournal#output#af392dc6-e445-4676-aff2-6af628469435#result]:
<https://nextjournal.com/data/QmZVEJ6LGCUyCz1F4K3iSFZmSY72DmBjgHExnHkgKCzhBy?content-type=image/svg%2Bxml&node-id=af392dc6-e445-4676-aff2-6af628469435&node-kind=output>

[nextjournal#output#5cbcbe39-2999-4038-aa24-ede86e1f9cb7#result]:
<https://nextjournal.com/data/QmTezq4At1fTsZ6cVWea1wPRbAttkhiQZKQ6E18p6xixXf?content-type=image/svg%2Bxml&node-id=5cbcbe39-2999-4038-aa24-ede86e1f9cb7&node-kind=output>

[nextjournal#output#28632111-2518-41e2-a685-3484234160e3#result]:
<https://nextjournal.com/data/QmVaCBiEhNNnJ25rkuBa296Akkpuk3rnrivgnidCtWqMV3?content-type=image/svg%2Bxml&node-id=28632111-2518-41e2-a685-3484234160e3&node-kind=output>

[nextjournal#output#b447f57a-1747-46c2-83e7-8115ac61afb2#result]:
<https://nextjournal.com/data/QmQDNCj3G6BTuxoq75KNnkyoaRafVmymGswGuJ5uvV4LYg?content-type=image/svg%2Bxml&node-id=b447f57a-1747-46c2-83e7-8115ac61afb2&node-kind=output>

[nextjournal#output#594974ba-13f3-4108-868a-faff9fbf63e3#result]:
<https://nextjournal.com/data/QmWvCXmvePxGihRHAuyebXouC3RW5yByEvHn4pdPKSHuZ6?content-type=image/svg%2Bxml&node-id=594974ba-13f3-4108-868a-faff9fbf63e3&node-kind=output>

[nextjournal#output#802593d5-3031-41a1-b6da-f89db94f4f2e#result]:
<https://nextjournal.com/data/QmcsKZUDSRj2k3dKRUVoDbShDsaGMC6fAN9UFYCnRPiRRx?content-type=image/svg%2Bxml&node-id=802593d5-3031-41a1-b6da-f89db94f4f2e&node-kind=output>

[nextjournal#file#6fdbe178-6447-492d-89a1-c032c78d864a]:
<https://nextjournal.com/data/QmNzpNwC6Dnhy2BvTfakQLWeWSA1GHjn3EVhQMamZHjdrb?content-type=image/jpeg&node-id=6fdbe178-6447-492d-89a1-c032c78d864a&filename=Filtro.jpg&node-kind=file>

[nextjournal#file#5aedebcc-628a-4134-9a5c-c5ef0de84372]:
<https://nextjournal.com/data/QmTFc5SqBWfJCxrVcY8hWRkVaDpSJ2wMoUopYfKoEojsbq?content-type=&node-id=5aedebcc-628a-4134-9a5c-c5ef0de84372&filename=filtroBp.npz&node-kind=file>

[nextjournal#reference#77f49149-5108-414f-9311-072688707cf8]:
<#nextjournal#reference#77f49149-5108-414f-9311-072688707cf8>

[nextjournal#reference#dbf7082d-2049-4494-a876-8ad6e01ad6b0]:
<#nextjournal#reference#dbf7082d-2049-4494-a876-8ad6e01ad6b0>

[nextjournal#output#d4e58430-754d-495c-90bf-f16d0ca72806#result]:
<https://nextjournal.com/data/QmfV47Wm9pSuUuH1PkHqdjWH6UHXoBXSyNiremYxfJyej3?content-type=image/svg%2Bxml&node-id=d4e58430-754d-495c-90bf-f16d0ca72806&node-kind=output>

[nextjournal#output#6e6e422b-d669-4bfc-9ba5-868ef747dc79#result]:
<https://nextjournal.com/data/QmWwYbwCZXqwtU57b8qRACGoQ5R5cvemj1hXKJUR3JS81P?content-type=image/svg%2Bxml&node-id=6e6e422b-d669-4bfc-9ba5-868ef747dc79&node-kind=output>

[nextjournal#output#c4b50706-6c14-4b84-b89d-9e753aee1a79#result]:
<https://nextjournal.com/data/QmP1eZqK6ECqa57XDaGHszjkfirWsVdVkpVSAeTJMu8EET?content-type=image/svg%2Bxml&node-id=c4b50706-6c14-4b84-b89d-9e753aee1a79&node-kind=output>

[nextjournal#file#dfe3a4a9-acda-497b-a190-2bb4a20b9f85]:
<https://nextjournal.com/data/QmbB7riUZ8pRpDYbkKTrh6j52huGvu7qmQZfK76bFmoU8U?content-type=image/jpeg&node-id=dfe3a4a9-acda-497b-a190-2bb4a20b9f85&filename=Polos_ceros.jpeg&node-kind=file>

[nextjournal#file#77ee5a75-1b76-4398-babe-3ed96705d60a]:
<https://nextjournal.com/data/QmRPPnRRbJkWtaK324PYdKsXG6ZjpPzKGgijsZFXfNGz1o?content-type=image/jpeg&node-id=77ee5a75-1b76-4398-babe-3ed96705d60a&filename=Fase.jpeg&node-kind=file>

<details id="com.nextjournal.article">
<summary>This notebook was exported from <a href="https://nextjournal.com/a/MrjE71icWXqPgwBqg7869?change-id=Cm3HrEn5FhFzboBaix3zfK">https://nextjournal.com/a/MrjE71icWXqPgwBqg7869?change-id=Cm3HrEn5FhFzboBaix3zfK</a></summary>

```edn nextjournal-metadata
{:article
 {:settings {:numbered? false},
  :nodes
  {"00f9cce1-407f-41f0-a89a-5df7b192c0c5"
   {:compute-ref #uuid "998c81ab-5703-4aec-b637-886cc5290b14",
    :exec-duration 34117,
    :id "00f9cce1-407f-41f0-a89a-5df7b192c0c5",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "042e9e2f-16f6-4652-829a-d6f44a61862e"
   {:compute-ref #uuid "93fd9a2d-b3de-4c39-ae2b-705216570286",
    :exec-duration 3276,
    :id "042e9e2f-16f6-4652-829a-d6f44a61862e",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "064bdb63-8d97-47eb-8ce1-23fa2fd8c824"
   {:compute-ref #uuid "cf6b07d0-5dcf-4d66-a9b3-58e39c3079ac",
    :exec-duration 681,
    :id "064bdb63-8d97-47eb-8ce1-23fa2fd8c824",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "0f7240a8-eb17-47a7-aab5-e4d1d8d44e7c"
   {:compute-ref #uuid "bd6909c3-9e68-4500-84d8-07af17032240",
    :exec-duration 999,
    :id "0f7240a8-eb17-47a7-aab5-e4d1d8d44e7c",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "140f871a-064f-45ff-8d29-a4e1ad0f10fe"
   {:compute-ref #uuid "426d8aa9-1e92-43b0-b554-8ac4e45aa633",
    :exec-duration 2207,
    :id "140f871a-064f-45ff-8d29-a4e1ad0f10fe",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "d218d0e6-1132-4ac3-98aa-d305bbb1fb82"]},
   "14b21fb6-2d54-4f4e-8591-a31999537f85"
   {:compute-ref #uuid "449eba21-99b7-48b2-a3fd-010eb50a2e86",
    :exec-duration 524,
    :id "14b21fb6-2d54-4f4e-8591-a31999537f85",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "1d662761-2f85-493c-bb8b-b5b834d22e49"
   {:id "1d662761-2f85-493c-bb8b-b5b834d22e49", :kind "file"},
   "25028c9c-3a6b-4623-875a-bc484beaaa57"
   {:compute-ref #uuid "0a103b10-4e87-434a-b770-74f10e290be5",
    :exec-duration 2331,
    :id "25028c9c-3a6b-4623-875a-bc484beaaa57",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "28632111-2518-41e2-a685-3484234160e3"
   {:compute-ref #uuid "d1fe1570-4fe4-4cd1-9d6e-5cd005abdbb4",
    :exec-duration 1426,
    :id "28632111-2518-41e2-a685-3484234160e3",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "2c220c20-317a-448d-a162-3c6742c9773f"
   {:compute-ref #uuid "ed9ad0b0-f4de-4661-a2fd-24cd158c4d44",
    :exec-duration 2272,
    :id "2c220c20-317a-448d-a162-3c6742c9773f",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "2e2671f2-1b8d-4f1d-aecc-545d1271c929"
   {:compute-ref #uuid "f31f06c7-0351-4fac-bf5c-5af668ad8907",
    :exec-duration 1068,
    :id "2e2671f2-1b8d-4f1d-aecc-545d1271c929",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "426d1501-3a13-436b-8626-baaa90743947"
   {:id "426d1501-3a13-436b-8626-baaa90743947", :kind "file"},
   "49d678c4-bc8e-425b-829f-263bc17bf9da"
   {:compute-ref #uuid "a42cd7bf-8a61-47d0-a53b-106341e92e69",
    :exec-duration 574,
    :id "49d678c4-bc8e-425b-829f-263bc17bf9da",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "4b40cba7-22f7-43d4-baa5-6222e3d7c085"
   {:environment
    [:environment
     {:article/nextjournal.id
      #uuid "02e6069e-f143-40cd-b284-f9ece1bf0a02",
      :change/nextjournal.id
      #uuid "5f203f65-f38c-4921-b1eb-970e41946c1c",
      :node/id "90436991-2923-44bb-b0d1-f235aec846ef"}],
    :id "4b40cba7-22f7-43d4-baa5-6222e3d7c085",
    :kind "runtime",
    :language "julia",
    :name "",
    :resources {:machine-type "n1-standard-4"},
    :type :nextjournal},
   "4bf5db09-fe71-41c7-8154-d3a5d04ef871"
   {:compute-ref #uuid "9c437e75-a9ea-4ecb-b66a-fecb8c6d9a22",
    :exec-duration 44268,
    :id "4bf5db09-fe71-41c7-8154-d3a5d04ef871",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "50f9c181-d22e-4149-87a3-fa865be97dad"
   {:compute-ref #uuid "7489d95d-2bba-4fe9-965e-bcc762cb6bb2",
    :exec-duration 485,
    :id "50f9c181-d22e-4149-87a3-fa865be97dad",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "558d8069-82ab-4023-80ff-94413a64f8ab"
   {:compute-ref #uuid "8aa96237-39ef-4141-ab1a-8653379e8c71",
    :exec-duration 542,
    :id "558d8069-82ab-4023-80ff-94413a64f8ab",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "575c5f5b-2957-4b1c-8683-001d4a8bc044"
   {:compute-ref #uuid "36746017-d095-4a64-ba1b-887a22f01a05",
    :exec-duration 1744,
    :id "575c5f5b-2957-4b1c-8683-001d4a8bc044",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "594974ba-13f3-4108-868a-faff9fbf63e3"
   {:compute-ref #uuid "0b08cec2-576d-4ff1-b3cd-61255f33a534",
    :exec-duration 1817,
    :id "594974ba-13f3-4108-868a-faff9fbf63e3",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "5aedebcc-628a-4134-9a5c-c5ef0de84372"
   {:id "5aedebcc-628a-4134-9a5c-c5ef0de84372", :kind "file"},
   "5cbcbe39-2999-4038-aa24-ede86e1f9cb7"
   {:compute-ref #uuid "24eb9c38-f3cc-4c40-b466-45e9c3b61b04",
    :exec-duration 210,
    :id "5cbcbe39-2999-4038-aa24-ede86e1f9cb7",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "6110fad1-f177-4d7e-ae38-da56153e339e"
   {:compute-ref #uuid "b8492944-3ae1-4263-8e7a-a6f777f5ee29",
    :exec-duration 2850,
    :id "6110fad1-f177-4d7e-ae38-da56153e339e",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "6e6e422b-d669-4bfc-9ba5-868ef747dc79"
   {:compute-ref #uuid "d2646da1-f7c8-445a-b2aa-705e5acc8369",
    :exec-duration 799,
    :id "6e6e422b-d669-4bfc-9ba5-868ef747dc79",
    :kind "code",
    :output-log-lines {:stdout 2},
    :runtime [:runtime "d218d0e6-1132-4ac3-98aa-d305bbb1fb82"]},
   "6fdbe178-6447-492d-89a1-c032c78d864a"
   {:id "6fdbe178-6447-492d-89a1-c032c78d864a", :kind "file"},
   "71237272-f2ba-4781-a4c4-f0efce96cfe5"
   {:id "71237272-f2ba-4781-a4c4-f0efce96cfe5", :kind "file"},
   "77e4bc62-63ab-413d-ba11-1c8f86326967"
   {:compute-ref #uuid "486f6dde-bfd5-463e-9301-042f68bcaf63",
    :exec-duration 613,
    :id "77e4bc62-63ab-413d-ba11-1c8f86326967",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "77ee5a75-1b76-4398-babe-3ed96705d60a"
   {:id "77ee5a75-1b76-4398-babe-3ed96705d60a", :kind "file"},
   "77f49149-5108-414f-9311-072688707cf8"
   {:id "77f49149-5108-414f-9311-072688707cf8",
    :kind "reference",
    :link [:output "fa3fd099-8ada-43ea-a176-770134676358" nil]},
   "802593d5-3031-41a1-b6da-f89db94f4f2e"
   {:compute-ref #uuid "22794361-ce2c-407c-8be2-f4f16588e3c6",
    :exec-duration 3303,
    :id "802593d5-3031-41a1-b6da-f89db94f4f2e",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "827fc4fe-e33d-45c0-acb4-c0bbdee00cfd"
   {:compute-ref #uuid "c7e8b0bb-e06f-43e5-bc81-056bf62d36db",
    :exec-duration 1240,
    :id "827fc4fe-e33d-45c0-acb4-c0bbdee00cfd",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "90d69ca0-3e5e-4430-8c1c-e56f9878b591"
   {:id "90d69ca0-3e5e-4430-8c1c-e56f9878b591", :kind "file"},
   "9ab3c8d0-c0a3-4c8b-b64d-ad4ce253f6c6"
   {:compute-ref #uuid "f42e3cb2-0a34-45c0-9f64-5cd10520439b",
    :exec-duration 7617,
    :id "9ab3c8d0-c0a3-4c8b-b64d-ad4ce253f6c6",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "9b97edd3-8ccf-4d87-82af-52cff4f513dc"
   {:compute-ref #uuid "7eda431d-eee1-4d71-98a1-17e2c64e89d8",
    :exec-duration 1861,
    :id "9b97edd3-8ccf-4d87-82af-52cff4f513dc",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "9df6d2a9-6d38-474b-88ea-667f5f938d8c"
   {:compute-ref #uuid "f70bbf08-7ba5-4f0b-b8dd-722493db4b8b",
    :exec-duration 509,
    :id "9df6d2a9-6d38-474b-88ea-667f5f938d8c",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "a49d697c-0d50-4142-a352-2a8d08419ab6"
   {:compute-ref #uuid "dd33589d-d622-49c8-831e-15ed52e1b1bf",
    :exec-duration 538,
    :id "a49d697c-0d50-4142-a352-2a8d08419ab6",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "af392dc6-e445-4676-aff2-6af628469435"
   {:compute-ref #uuid "4f35022f-4469-44fb-abb0-99b4daa3e992",
    :exec-duration 331,
    :id "af392dc6-e445-4676-aff2-6af628469435",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "b447f57a-1747-46c2-83e7-8115ac61afb2"
   {:compute-ref #uuid "a0ecaf13-4ef6-4235-8ea2-d09b35b7afc0",
    :exec-duration 1751,
    :id "b447f57a-1747-46c2-83e7-8115ac61afb2",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "b8f245ea-096c-4994-9e53-2f90ace5dcdb"
   {:compute-ref #uuid "c0f1e988-eec6-4f33-a3c3-8a6fd669b24a",
    :exec-duration 770,
    :id "b8f245ea-096c-4994-9e53-2f90ace5dcdb",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "bb4c58bd-1bf1-406a-a857-e0bd58b42c39"
   {:id "bb4c58bd-1bf1-406a-a857-e0bd58b42c39",
    :kind "reference",
    :link [:output "fa3fd099-8ada-43ea-a176-770134676358" nil]},
   "bf44e8c3-bd93-4926-8b46-688516d55d83"
   {:compute-ref #uuid "b23c2c74-f2c3-45b8-9c4e-e24e78c9ec0c",
    :exec-duration 789,
    :id "bf44e8c3-bd93-4926-8b46-688516d55d83",
    :kind "code",
    :output-log-lines {:stdout 226},
    :runtime [:runtime "d218d0e6-1132-4ac3-98aa-d305bbb1fb82"],
    :stdout-collapsed? false},
   "c10a69bb-8e59-493f-bb02-2db74abf6e2f"
   {:compute-ref #uuid "b7a41408-c557-4441-942c-bdebf066dde5",
    :exec-duration 180,
    :id "c10a69bb-8e59-493f-bb02-2db74abf6e2f",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "c4b50706-6c14-4b84-b89d-9e753aee1a79"
   {:compute-ref #uuid "ba3aba5f-3ceb-42f7-bd4d-57c298c9ffb1",
    :exec-duration 717,
    :id "c4b50706-6c14-4b84-b89d-9e753aee1a79",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "d218d0e6-1132-4ac3-98aa-d305bbb1fb82"
   {:environment
    [:environment
     {:article/nextjournal.id
      #uuid "5b45e08b-5b96-413e-84ed-f03b5b65bd66",
      :change/nextjournal.id
      #uuid "5f0c0e79-790f-439a-8b18-fb81409f12c2",
      :node/id "0149f12a-08de-4f3d-9fd3-4b7a665e8624"}],
    :id "d218d0e6-1132-4ac3-98aa-d305bbb1fb82",
    :kind "runtime",
    :language "python",
    :type :nextjournal},
   "d4e58430-754d-495c-90bf-f16d0ca72806"
   {:compute-ref #uuid "e58c417e-851f-42cd-ab69-6b948fffd04e",
    :exec-duration 739,
    :id "d4e58430-754d-495c-90bf-f16d0ca72806",
    :kind "code",
    :output-log-lines {:stdout 2},
    :runtime [:runtime "d218d0e6-1132-4ac3-98aa-d305bbb1fb82"],
    :stdout-collapsed? false},
   "d7218fa5-abe0-419f-884c-75df324c0184"
   {:id "d7218fa5-abe0-419f-884c-75df324c0184", :kind "file"},
   "d849cb78-0489-4974-a8c9-90e3780bad30"
   {:compute-ref #uuid "31ec6e8e-1703-4a4e-ac9c-f9ee78ec6194",
    :exec-duration 1543,
    :id "d849cb78-0489-4974-a8c9-90e3780bad30",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "dbf7082d-2049-4494-a876-8ad6e01ad6b0"
   {:id "dbf7082d-2049-4494-a876-8ad6e01ad6b0",
    :kind "reference",
    :link [:output "5aedebcc-628a-4134-9a5c-c5ef0de84372" nil]},
   "dfe3a4a9-acda-497b-a190-2bb4a20b9f85"
   {:id "dfe3a4a9-acda-497b-a190-2bb4a20b9f85", :kind "file"},
   "e1c3e588-a573-42a7-b123-606ad7686ece"
   {:compute-ref #uuid "ee5ccd53-ea93-4fa4-8e48-a82e6ebc058c",
    :exec-duration 590,
    :id "e1c3e588-a573-42a7-b123-606ad7686ece",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "e3dc35c2-bde3-41c8-b38b-b8d8114b6167"
   {:compute-ref #uuid "e9ff5ad0-dd5b-4f91-be01-922b11424cfa",
    :exec-duration 1638,
    :id "e3dc35c2-bde3-41c8-b38b-b8d8114b6167",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "ec4eedb4-5b29-4ab4-bd25-507ce4b635be"
   {:compute-ref #uuid "b8df51e6-3bdc-4fca-93eb-ec142c14566e",
    :exec-duration 504,
    :id "ec4eedb4-5b29-4ab4-bd25-507ce4b635be",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "f3d6773e-3683-4f83-9472-a47bd0994ab9"
   {:compute-ref #uuid "746b7121-8933-4d6f-ac12-ff4a006d6df5",
    :exec-duration 383,
    :id "f3d6773e-3683-4f83-9472-a47bd0994ab9",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "4b40cba7-22f7-43d4-baa5-6222e3d7c085"]},
   "fa3fd099-8ada-43ea-a176-770134676358"
   {:id "fa3fd099-8ada-43ea-a176-770134676358", :kind "file"}},
  :nextjournal/id #uuid "02e98579-c54e-439c-83c9-497e18ec7e06",
  :article/change
  {:nextjournal/id #uuid "5f3aa9a7-5a0a-427c-8c79-5dc597d5dc52"}}}

```
</details>
