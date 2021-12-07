g=sin([1:0.1:10*pi]);
for i = 1:length(g)
  figure(1)    
    if i ~=length(g)        
        plot(1:i,g(1,1:i),'-b');      
    else
        plot(1:i,g(1,1:i),'-b*')
    end
    if i>=50
        axis([i-50 i+50 min(g(1,:)) max(g(1,:))])
    else
        axis([0 i+50 min(g(1,:)) max(g(1,:))])
    end
 %pause(0.1)
    drawnow
end
