function R = rotx(theta)
    % theta in degrees (consistent with MATLAB rotx)
    ct = cosd(theta);
    st = sind(theta);
    R = [1 0  0;
         0 ct -st;
         0 st  ct];
end